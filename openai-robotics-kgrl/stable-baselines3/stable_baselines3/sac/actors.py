import warnings
from typing import Any, Dict, List, Optional, Tuple, Type

import gym
import torch as th
from torch import nn

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp

import pdb

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(BasePolicy): 
    """
    Base actor network (policy).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super(Actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)

class RLActor(Actor):
    """
    Actor network (policy) for SAC.

    :param kg_params: parameters related to external knowledge, None for RLActor
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        kg_params: dict = None, #RL
    ):
        super(RLActor, self).__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            clip_mean,
            normalize_images,
        )

        action_dim = get_action_dim(self.action_space)
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)

        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

class KIANActor(Actor):
    """
    Actor network (policy) for SAC.

    :param kg_params: parameters related to external knowledge
        kg_num: number of external knowledge policies
        env_type: environment type for external knowledge selection
        kg_log_std: log std of external knowledge policies
        kg_emb_dim: dimension of a knowledge embedding
        kg_hid_num: number of attention network's hidden layers
        kg_hid_dim: dimension of each attention network's hidden layer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        kg_params: dict = None, #KIAN
    ):
        super(KIANActor, self).__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            clip_mean,
            normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        #-----------#
        #KIAN starts#
        #-----------#

        self.kg_params = kg_params

        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)

        #---------#
        #KIAN ends#
        #---------#

        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            #-----------#
            #KIAN starts#
            #-----------#

            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.k_action_dist = DiagGaussianDistribution(action_dim)

            self.kg_num = self.kg_params['kg_num']
            self.env_type = self.kg_params['env_type']
            self.kg_log_std = self.kg_params['kg_log_std']
            kg_emb_dim = self.kg_params['kg_emb_dim']
            kg_hid_num = self.kg_params['kg_hid_num']
            kg_hid_dim = self.kg_params['kg_hid_dim']
            get_kg_func = self.kg_params['get_kg_func']

            print('kg_num: ', self.kg_num)
            print('env_type: ', self.env_type)
            print('kg_log_std: ', self.kg_log_std)
            print('kg_emb_dim: ', kg_emb_dim)
            print('kg_hid_num: ', kg_hid_num)
            print('kg_hid_dim: ', kg_hid_dim)
            print('get_kg_func: ', get_kg_func)

            # query network
            query_net_arch = [kg_hid_dim for _ in range(kg_hid_num)]
            query_net = create_mlp(features_dim, kg_emb_dim, query_net_arch, nn.ReLU)
            self.actor_Q = nn.Sequential(*query_net)

            # inner key network
            self.actor_K = nn.Embedding(1, kg_emb_dim)

            # inner policy
            self.mu_V = nn.Linear(last_layer_dim, action_dim)
            self.log_std_V = nn.Linear(last_layer_dim, action_dim)

            # rules embeddings
            self.kg_K = nn.Embedding(self.kg_num, kg_emb_dim)

            self.action_dim = action_dim

            # import external knowledge policies
            exec('from stable_baselines3.common.env_kg import {}'.format(get_kg_func))
            exec('self.getKGAction = {}'.format(get_kg_func))

            #---------#
            #KIAN ends#
            #---------#

    #-----------#
    #KIAN starts#
    #-----------#

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.
        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs)
        latent_pi = self.latent_pi(features)

        actor_mean_actions = self.mu_V(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        actor_log_std = self.log_std_V(latent_pi) #(batch size, action dim)
        actor_log_std = th.clamp(actor_log_std, LOG_STD_MIN, LOG_STD_MAX)

        actor_mean_actions = actor_mean_actions.unsqueeze(1)
        actor_log_std = actor_log_std.unsqueeze(1)

        # Get KG actions
        # (batch size, kg num, action dim), (batch size, kg num, action dim)
        kg_mean_actions, kg_log_std = self.getKGAction(
            obs, 
            self.action_dim, 
            self.env_type, 
            self.kg_log_std, 
            self.device,
        )

        # Calculate weights for actor/KG actions
        # w_actor: (1 x 1 x kg emb dim) x (batch size x kg emb dim x 1)
        w_actor = th.matmul(self.actor_K.weight.unsqueeze(0), self.actor_Q(features).unsqueeze(2))
        # w_kg: (1 x kg num  x kg emb dim) x (batch size x kg emb dim x 1)
        w_kg = th.matmul(self.kg_K.weight.unsqueeze(0), self.actor_Q(features).unsqueeze(2))
        weights = th.concat([w_kg, w_actor], dim=1) #batch size x (kg num + 1) x 1

        # Combine actions and log_stds
        mean_actions = th.concat([kg_mean_actions,actor_mean_actions],dim=1)
        log_stds = th.concat([kg_log_std,actor_log_std],dim=1)

        return mean_actions, log_stds, weights, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_stds, weights, kwargs = self.get_action_dist_params(obs)

        # Note: the action is squashed
        in_act = self.action_dist.actions_from_params(mean_actions[:,-1,:], log_stds[:,-1,:], deterministic=deterministic, **kwargs)

        kg_act = mean_actions[:,:-1,:]
        acts = th.cat([kg_act, in_act.unsqueeze(1)], dim=1) #batch size x (kg num+1) x action dim

        w_sample = th.nn.functional.gumbel_softmax(weights[:,:,0], hard=True) #batch size x (kg num+1)

        return th.sum(acts * w_sample.unsqueeze(-1), dim=1) #batch size x action dim

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_stds, weights, kwargs = self.get_action_dist_params(obs)

        # inner actor's action and associated log prob
        # (batch size, action dim), (batch size,)
        in_act, in_log_prob = self.action_dist.log_prob_from_params(mean_actions[:,-1,:], log_stds[:,-1,:], **kwargs)
        # prevent overflow after exp() and log()
        in_log_prob = th.clamp(in_log_prob, min=-18)

        # external knowledge's action
        kg_act = mean_actions[:,:-1,:]

        # sample an action from the Gaussian mixture distribution using Gumbel softmax
        w_sample = th.nn.functional.gumbel_softmax(weights[:,:,0], hard=True) #batch size x (kg num+1)
        acts = th.cat([kg_act, in_act.unsqueeze(1)], dim=1) #batch size x (kg num+1) x action dim
        actions = th.sum(acts * w_sample.unsqueeze(-1), dim=1) #batch size x action dim

        # log prob of the chosen action in knowledge policy's probability distribution
        kg_log_prob = []
        for idx in range(0, self.kg_num): 
            self.k_action_dist.proba_distribution(mean_actions[:,idx,:], log_stds[:,idx,:])
            # append batch size x 1 
            kg_log_prob.append(self.k_action_dist.log_prob(mean_actions[:,idx,:]).unsqueeze(1))
        kg_log_prob = th.cat(kg_log_prob, dim=1) #batch size x kg num
        # prevent overflow after exp() and log()
        kg_log_prob = th.clamp(kg_log_prob, min=-18)

        weights_softmax = th.nn.functional.softmax(weights, dim=1) #batch size x (kg num+1) x 1
        probs = th.exp(th.cat([kg_log_prob, in_log_prob.unsqueeze(1)], dim=1)) #batch size x (kg num+1)
        log_probs = th.log(th.sum(weights_softmax[:,:,0] * probs, dim=1)) #(batch size,)

        return (actions, log_probs)
