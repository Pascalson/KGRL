import torch
import torch.nn.functional as F

def get_kg_set(env_name):
    if env_name in ["MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-Random-5x5-v0", "MiniGrid-Empty-16x16-v0"]:
        kg_set = [
            "go to the goal",
        ]
    elif env_name in ["MiniGrid-Unlock-v0"]:
        kg_set = [
            "get the key",
            "open the door",
        ]
    elif env_name in ["MiniGrid-DoorKey-5x5-v0", "MiniGrid-DoorKey-8x8-v0"]:
        kg_set = [
            "get the key",
            "open the door",
            "go to the goal",
        ]
    elif env_name in ["MiniGrid-Dynamic-Obstacles-Random-6x6-v0", "MiniGrid-Dynamic-Obstacles-16x16-v0"]:
        kg_set = [
            "go to the goal",
            "do not hit ball",
        ]
    elif env_name in ["MiniGrid-LavaCrossingS9N2-v0"]:
        kg_set = [
            "go to the goal",
            "do not hit",
        ]
    elif "KeyCorridor" in env_name:
        kg_set = [
            "open one unlocked door",
            "open the locked door",
            "get the key",
            "pick up the ball",
        ]
    elif env_name in ["MiniGrid-MultiRoom-N4-S5-v0"]:
        kg_set = [
            "open the unlocked door",
            "go to the goal",
            "do not hit",
        ]
    elif env_name == 'all':
        kg_set = [
            "get the key",
            "open the door",
            "go to the goal",
            "do not hit ball",
            "do not hit",
            "open the unlocked door",
        ]
    return kg_set


actions = {
    "Left":0,
    "Right":1,
    "Forward":2,
    "Pickup":3,
    "Drop":4,
    "Toggle":5,
    "Done":6,
}

def get_expert_actions(obs, expert_rules, env_name=""):
    agent_pos = (obs.image.shape[1]//2, obs.image.shape[2]-1)
    expert_actions = torch.zeros((obs.image.shape[0], len(expert_rules), len(actions)), device='cuda:0')
    img = obs.image[:,:,:,0]
    colors = obs.image[:,:,:,1]
    states = obs.image[:,:,:,2]
    
    def convert_pos_to_dir_actions(rule_id, start, goals, ids_to_remove=None):
        img_id_meet_condition = goals[:,0][(start[0] < goals[:,1]).int().nonzero()]
        expert_actions[img_id_meet_condition,rule_id,actions["Right"]] = 1
        img_id_meet_condition = goals[:,0][(start[0] > goals[:,1]).int().nonzero()]
        expert_actions[img_id_meet_condition,rule_id,actions["Left"]] = 1
        img_id_meet_condition = goals[:,0][(start[1] > goals[:,2]).int().nonzero()]
        expert_actions[img_id_meet_condition,rule_id,actions["Forward"]] = 1
        if ids_to_remove is not None:
            expert_actions[ids_to_remove,rule_id,actions["Right"]] = 0
            expert_actions[ids_to_remove,rule_id,actions["Left"]] = 0
            expert_actions[ids_to_remove,rule_id,actions["Forward"]] = 0

    def prevent_actions(rule_id, start, goals):
        expert_actions[:,rule_id,actions["Right"]] = 1
        expert_actions[:,rule_id,actions["Left"]] = 1
        expert_actions[:,rule_id,actions["Forward"]] = 1
        img_id_meet_condition = goals[:,0][torch.logical_and(start[0] - goals[:,1] == -1, start[1] == goals[:,2]).int().nonzero()]
        expert_actions[img_id_meet_condition,rule_id,actions["Right"]] = 0
        img_id_meet_condition = goals[:,0][torch.logical_and(start[0] - goals[:,1] == 1,  start[1] == goals[:,2]).int().nonzero()]
        expert_actions[img_id_meet_condition,rule_id,actions["Left"]] = 0
        img_id_meet_condition = goals[:,0][torch.logical_and(start[1] - goals[:,2] == 1,  start[0] == goals[:,1]).int().nonzero()]
        expert_actions[img_id_meet_condition,rule_id,actions["Forward"]] = 0

    # Knowledge: get the key, id=5
    if "get the key" in expert_rules:
        rule_id = expert_rules.index("get the key")
        key_pos = (img == 5).nonzero()
        img_id_meet_condition = key_pos[:,0][((key_pos[:,1] == agent_pos[0]).int() * (key_pos[:,2] == agent_pos[1]-1).int()).nonzero()]
        expert_actions[img_id_meet_condition,rule_id,actions["Pickup"]] = 1
        convert_pos_to_dir_actions(rule_id, agent_pos, key_pos, ids_to_remove=img_id_meet_condition)

    # Knowledge: pick up the ball, id=6
    if "pick up the ball" in expert_rules:
        rule_id = expert_rules.index("pick up the ball")
        key_pos = (img == 6).nonzero()
        img_id_meet_condition = key_pos[:,0][((key_pos[:,1] == agent_pos[0]).int() * (key_pos[:,2] == agent_pos[1]-1).int()).nonzero()]
        expert_actions[img_id_meet_condition,rule_id,actions["Pickup"]] = 1
        convert_pos_to_dir_actions(rule_id, agent_pos, key_pos, ids_to_remove=img_id_meet_condition)

    # Knowledge: open the door, id=4
    if "open the door" in expert_rules:
        rule_id = expert_rules.index("open the door")
        door_pos = (img == 4).nonzero()
        img_id_meet_condition = door_pos[:,0][((door_pos[:,1] == agent_pos[0]).int() * (door_pos[:,2] == agent_pos[1]-1).int()).nonzero()]
        expert_actions[img_id_meet_condition,rule_id,actions["Toggle"]] = 1
        convert_pos_to_dir_actions(rule_id, agent_pos, door_pos, ids_to_remove=img_id_meet_condition)

    # Knowledge: go to the goal, id=8
    if "go to the goal" in expert_rules:
        rule_id = expert_rules.index("go to the goal")
        goal_pos = (img == 8).nonzero()
        convert_pos_to_dir_actions(rule_id, agent_pos, goal_pos)

    # Knowledge: do not hit, wall id=2; lava id=9
    if "do not hit" in expert_rules:
        rule_id = expert_rules.index("do not hit")
        prevent_pos = torch.logical_or(img == 2,img ==9).nonzero()
        prevent_actions(rule_id, agent_pos, prevent_pos)

    # Knowledge: do not hit ball, id=6 # NOTE: can merge into "do not hit" knowledge.
    if "do not hit ball" in expert_rules:
        rule_id = expert_rules.index("do not hit ball")
        prevent_pos = (img == 6).nonzero()
        prevent_actions(rule_id, agent_pos, prevent_pos)

    # Knowledge: go through a wall or lava, wall id=2; lava id=9 # NOTE: unused
    if "go through wall" in expert_rules:
        rule_id = expert_rules.index("go through wall")
        prevent_pos = torch.logical_or(img == 2,img ==9).nonzero()
        prevent_actions(rule_id, agent_pos, prevent_pos)

    # Knowledge: open the locked door, id=4, state=2
    if "open the locked door" in expert_rules:
        rule_id = expert_rules.index("open the locked door")
        door_pos = torch.logical_and(img==4,states==2).nonzero()
        img_id_meet_condition = door_pos[:,0][((door_pos[:,1] == agent_pos[0]).int() * (door_pos[:,2] == agent_pos[1]-1).int()).nonzero()]
        expert_actions[img_id_meet_condition,rule_id,actions["Toggle"]] = 1
        convert_pos_to_dir_actions(rule_id, agent_pos, door_pos, ids_to_remove=img_id_meet_condition)

    # Knowledge: open the unlocked door, id=4, state=1
    if "open the unlocked door" in expert_rules:
        rule_id = expert_rules.index("open the unlocked door")
        door_pos = torch.logical_and(img==4,states==1).nonzero()
        img_id_meet_condition = door_pos[:,0][((door_pos[:,1] == agent_pos[0]).int() * (door_pos[:,2] == agent_pos[1]-1).int()).nonzero()]
        expert_actions[img_id_meet_condition,rule_id,actions["Toggle"]] = 1
        convert_pos_to_dir_actions(rule_id, agent_pos, door_pos, ids_to_remove=img_id_meet_condition)

    # Knowledge: open only one unlocked door, id=4, state=1
    if "open one unlocked door" in expert_rules:
        rule_id = expert_rules.index("open one unlocked door")
        door_pos = torch.logical_and(img==4,states==1).nonzero()
        if door_pos.shape[0] > 1:
            door_pos = torch.cat([door_pos[0].view(1,-1), door_pos[1:][(door_pos[1:,0] - door_pos[:-1,0] != 0)]],dim=0)
        img_id_meet_condition = door_pos[:,0][((door_pos[:,1] == agent_pos[0]).int() * (door_pos[:,2] == agent_pos[1]-1).int()).nonzero()]
        expert_actions[img_id_meet_condition,rule_id,actions["Toggle"]] = 1
        convert_pos_to_dir_actions(rule_id, agent_pos, door_pos, ids_to_remove=img_id_meet_condition)

    if "Dynamic-Obstacles" in env_name:# change the action space to include only three actions, the requirement of that environment
        expert_actions = expert_actions[:,:,:3]

    return expert_actions

def expert_behaviors_by_env(env_name='all'):
    expert_rules = get_kg_set(env_name)
    def expert_behaviors(obs):
        expert_actions = get_expert_actions(obs, expert_rules, env_name=env_name)# (b x # rules x # actions) or (# rules x # actions)
        x = torch.sum(expert_actions, dim=1) #(b x # actions)
        return F.softmax(x, dim=1)
    return lambda x: expert_behaviors(x)
