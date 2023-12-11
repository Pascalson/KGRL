import numpy as np
import torch as th

import pdb

def fetchGetKGAction(
        obs: th.Tensor, 
        action_dim: int, 
        env_type: str, 
        log_std_min: int, 
        device: th.device, 
    ): 
    """
    Get the action inferred from the knowledge for Fetch environments
    param obs: observations (Dict, {'observation', 'achieved_goal', 'desired_goal'}) (batch size x ?)
    param action_dim: dimension of an action
    param env_type: type of Fetch environment
    param log_std_min: min value of log std
    param device: device where the outputs are stored
    return kg_mean_actions: mean actions from the predefined knowledge (batch size x # of knowledge x action dim)
    return kg_log_std: log std from the predefined knowledge (batch size x # of knowledge x action dim)
    """

    MAX_AC = 1.

    batch_size = obs['observation'].shape[0]
    desired_goal = obs['desired_goal']
    current_grip_pos = obs['observation'][:,:3]

    kg_num = 2
    kg_mean_actions = th.zeros([batch_size, kg_num, action_dim])

    if 'PickAndPlace' in env_type: 
        rel_pos, g_state = obs['observation'][:,6:9], obs['observation'][:,9].unsqueeze(1)
        rel_pos_norm = th.linalg.norm(rel_pos, dim=-1, keepdim=True) #batch size x 1

        #first knowledge: move to object
        kg_mean_actions[:,0,:3] = rel_pos * (rel_pos_norm >= 0.03).int()
        kg_mean_actions[:,0,-1] = (1. * (rel_pos_norm >= 0.03).int() + g_state * (rel_pos_norm < 0.03).int())[:,0]

        #second knowledge: move to target
        kg_mean_actions[:,1,:3] = (desired_goal - current_grip_pos) * (rel_pos_norm < 0.03).int()
        kg_mean_actions[:,1,-1] = (-1. * (rel_pos_norm < 0.03).int() + g_state * (rel_pos_norm >= 0.03).int())[:,0]

    elif 'Push' in env_type: 
        #gripper is locked
        rel_pos = obs['observation'][:,6:9]
        rel_pos_norm = th.linalg.norm(rel_pos, dim=-1, keepdim=True) #batch size x 1

        #first knowledge: move to object
        kg_mean_actions[:,0,:3] = rel_pos * (rel_pos_norm >= 0.03).int()

        #second knowledge: move to target
        kg_mean_actions[:,1,:3] = (desired_goal - current_grip_pos) * (rel_pos_norm < 0.03).int()

    elif 'Slide' in env_type: 
        #gripper is locked
        rel_pos = obs['observation'][:,6:9]
        rel_pos_norm = th.linalg.norm(rel_pos, dim=-1, keepdim=True) #batch size x 1

        #first knowledge: move to intermediate goal (somewhere behind object)
        obj_goal = obs['observation'][:,3:6] - desired_goal #batch size x 3
        obj_goal_distance = th.linalg.norm(obj_goal, dim=-1, keepdim=True) #batch size x 1
        intermediate_goal = desired_goal + obj_goal * (1 + 0.1 / (obj_goal_distance+1e-6))
        int_goal_rel_pos = intermediate_goal - current_grip_pos
        int_goal_rel_pos_norm = th.linalg.norm(int_goal_rel_pos, dim=-1, keepdim=True) #batch size x 1
        kg_mean_actions[:,0,:3] = int_goal_rel_pos * (int_goal_rel_pos_norm >= 0.03).int()

        #second knowledge: move to target
        kg_mean_actions[:,1,:3] = (desired_goal - current_grip_pos) * (int_goal_rel_pos_norm < 0.03).int()

    kg_mean_actions = _scaleActions(kg_mean_actions, MAX_AC)
    kg_log_std = th.ones([batch_size, kg_num, action_dim]) * log_std_min

    return kg_mean_actions.to(device), kg_log_std.to(device)

def _scaleActions(actions, max_ac): 
    abs_actions = th.abs(actions)
    max_abs_value = th.unsqueeze(th.amax(abs_actions, dim=-1), -1)
    max_abs_value = th.maximum(th.ones_like(max_abs_value), max_abs_value)
    actions = max_ac / max_abs_value * actions

    return actions
