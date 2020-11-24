from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch

from utils.additional import storage_saver


class GAE:
    """ Generalized Advantage Estimator.
    See [Schulman et al., 2016](https://arxiv.org/abs/1506.02438)
    """

    def __init__(self, policy, gamma=0.99, lambda_=0.95, normalize=None,
                 epsilon=1e-8):
        self.policy = policy
        self.gamma = gamma
        self.lambda_ = lambda_
        self.normalize = normalize
        self.epsilon = epsilon

    def __call__(self, trajectory):
        """ Applies the advantage estimator to a given trajectory.
        Returns:
          a tuple of (advantages, value_targets).
        """
        if "advantages" in trajectory:
            raise ValueError("trajectory cannot contain 'advantages'")
        if "value_targets" in trajectory:
            raise ValueError("trajectory cannot contain 'value_targets'")

        rewards = trajectory["rewards"]
        resets = trajectory["resets"]
        values = trajectory["values"]

        # Values might have an additional last dimension of size 1 as outputs of
        # dense layers. Need to adjust shapes of rewards and resets accordingly.
        if (not (0 <= values.ndim - rewards.ndim <= 1)
                or values.ndim == rewards.ndim + 1 and values.shape[-1] != 1):
            raise ValueError(
                f"trajectory['values'] of shape {trajectory['values'].shape} "
                "must have the same number of dimensions as "
                f"trajectory['rewards'] which has shape {rewards.shape} "
                "or have last dimension of size 1")
        if values.ndim == rewards.ndim + 1:
            values = np.squeeze(values, -1)

        gae = np.zeros_like(values, dtype=np.float32)
        gae[-1] = rewards[-1] - values[-1]
        observation = trajectory["state"]["latest_observation"]
        state = trajectory["state"].get("policy_state", None)
        last_value = self.policy.act(observation, state=state,
                                     update_state=False)["values"]
        if np.asarray(resets[-1]).ndim < last_value.ndim:
            last_value = np.squeeze(last_value, -1)
        gae[-1] += (1 - resets[-1]) * self.gamma * last_value

        for i in range(gae.shape[0] - 1, 0, -1):
            not_reset = 1 - resets[i - 1]
            next_values = values[i]
            delta = (rewards[i - 1]
                     + not_reset * self.gamma * next_values
                     - values[i - 1])
            gae[i - 1] = delta + not_reset * self.gamma * self.lambda_ * gae[i]
        value_targets = gae + values
        value_targets = value_targets[
            (...,) + (None,) * (trajectory["values"].ndim - value_targets.ndim)]

        if self.normalize or self.normalize is None and gae.size > 1:
            gae = (gae - gae.mean()) / (gae.std() + self.epsilon)

        trajectory["advantages"] = gae
        trajectory["value_targets"] = value_targets
        return gae, value_targets


class BaseRunner1(ABC):
    """ General data runner. """

    def __init__(self, env, policy, step_var=None):
        self.env = env
        self.policy = policy
        if step_var is None:
            step_var = torch.zeros(1, requires_grad=False)
        self.step_var = step_var

    @property
    def nenvs(self):
        """ Returns number of batched envs or `None` if env is not batched """
        return getattr(self.env.unwrapped, "nenvs", None)

    @abstractmethod
    def get_next(self):
        """ Returns next data object """


class BaseRunnerNoController(ABC):
    """ General data runner. """

    def __init__(self, env, step_var=None):
        self.env = env
        if step_var is None:
            step_var = torch.zeros(1, requires_grad=False)
        self.step_var = step_var

    @property
    def nenvs(self):
        """ Returns number of batched envs or `None` if env is not batched """
        return getattr(self.env.unwrapped, "nenvs", None)

    @abstractmethod
    def get_next(self):
        """ Returns next data object """


class EnvRunnerNoController(BaseRunnerNoController):
    # pylint: disable=too-many-arguments
    def __init__(self, env, nsteps, cutoff=None, asarray=True, transforms=None, step_var=None):
        super().__init__(env, step_var)
        self.env = env
        self.nsteps = nsteps
        self.cutoff = cutoff
        self.asarray = asarray
        self.transforms = transforms or []
        self.state = {"latest_observation": self.env.reset()}

    def reset(self, policy=None):
        """ Resets env and runner states. """
        self.state["latest_observation"] = self.env.reset()
        if policy:
            policy.reset()

    def get_next(self, act, observations, rewards, resets, trajectory):
        """ Runs the agent in the environment.  """
        observations.append(self.state["latest_observation"])
        if "actions" not in act:
            raise ValueError(f"result of policy.act must contain 'actions' but has keys {list(act.keys())}")
        for key, val in act.items():
            if isinstance(trajectory[key], list):
                trajectory[key].append(val)
            if isinstance(trajectory[key], np.ndarray):
                trajectory[key] = trajectory[key].tolist()
                trajectory[key].append(val)

        architecture = trajectory["actions"][-1]
        storage_saver.set_architecture(architecture)
        obs, rew, done, _ = self.env.step(architecture)
        self.state["latest_observation"] = obs
        rewards.append(rew)
        resets.append(done)
        # self.step_var.assign_add(self.nenvs or 1)
        if self.nenvs is not None and self.nenvs > 1:
            self.step_var += self.nenvs
        else:
            self.step_var += 1

        # Only reset if the env is not batched. Batched envs should auto-reset.
        if not self.nenvs and np.all(done):
            self.state["env_steps"] = 1
            self.state["latest_observation"] = self.env.reset()

        trajectory.update(observations=observations, rewards=rewards, resets=resets)

        if self.asarray:
            for key, val in trajectory.items():
                try:
                    trajectory[key] = np.asarray(val)
                except ValueError:
                    raise ValueError(f"cannot convert value under key '{key}' to np.ndarray")
        trajectory["state"] = self.state

        for transform in self.transforms:
            transform(trajectory)

        return trajectory


class EnvRunner1(BaseRunner1):
    # pylint: disable=too-many-arguments
    def __init__(self, env, policy, nsteps, cutoff=None, asarray=True, transforms=None, step_var=None):
        super().__init__(env, policy, step_var)
        self.env = env
        self.policy = policy
        self.nsteps = nsteps
        self.cutoff = cutoff
        self.asarray = asarray
        self.transforms = transforms or []
        self.state = {"latest_observation": self.env.reset()}

    def reset(self):
        """ Resets env and runner states. """
        self.state["latest_observation"] = self.env.reset()
        self.policy.reset()

    def get_next(self):
        """ Runs the agent in the environment.  """
        trajectory = defaultdict(list, {"actions": []})
        observations = []
        rewards = []
        resets = []
        self.state["env_steps"] = self.nsteps
        if self.policy.is_recurrent():
            self.state["policy_state"] = self.policy.get_state()

        for i in range(self.nsteps):
            observations.append(self.state["latest_observation"])
            act = self.policy.act(self.state["latest_observation"])
            if "actions" not in act:
                raise ValueError(f"result of policy.act must contain 'actions' but has keys {list(act.keys())}")
            for key, val in act.items():
                trajectory[key].append(val)
            obs, rew, done, _ = self.env.step(trajectory["actions"][-1])
            self.state["latest_observation"] = obs
            rewards.append(rew)
            resets.append(done)
            if self.nenvs is not None and self.nenvs > 1:
                self.step_var += self.nenvs
            else:
                self.step_var += 1

            # Only reset if the env is not batched. Batched envs should auto-reset.
            if not self.nenvs and np.all(done):
                self.state["env_steps"] = i + 1
                self.state["latest_observation"] = self.env.reset()
                if self.cutoff or (self.cutoff is None and self.policy.is_recurrent()):
                    break

        trajectory.update(observations=observations, rewards=rewards, resets=resets)
        if self.asarray:
            for key, val in trajectory.items():
                try:
                    trajectory[key] = np.asarray(val)
                except ValueError:
                    raise ValueError(f"cannot convert value under key '{key}' to np.ndarray")
        trajectory["state"] = self.state

        for transform in self.transforms:
            transform(trajectory)
        return trajectory


class TrajectorySampler(BaseRunner1):
    """ Samples parts of trajectory for specified number of epochs. """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, runner, num_epochs=4, num_minibatches=4,
                 shuffle_before_epoch=True, transforms=None):
        super().__init__(runner.env, runner.policy, runner.step_var)
        self.runner = runner
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.shuffle_before_epoch = shuffle_before_epoch
        self.transforms = transforms or []
        self.minibatch_count = 0
        self.epoch_count = 0
        self.trajectory = None

    def trajectory_is_stale(self):
        """ True iff new trajectory should be generated for sub-sampling. """
        return self.epoch_count >= self.num_epochs

    def shuffle_trajectory(self):
        """ Reshuffles trajectory along the first dimension. """
        sample_size = self.trajectory["observations"].shape[0]
        indices = np.random.permutation(sample_size)
        for key, val in filter(lambda kv: isinstance(kv[1], np.ndarray),
                               self.trajectory.items()):
            self.trajectory[key] = val[indices]

    def get_next(self):
        if self.trajectory is None or self.trajectory_is_stale():
            self.epoch_count = self.minibatch_count = 0
            self.trajectory = self.runner.get_next()
            if self.shuffle_before_epoch:
                self.shuffle_trajectory()

        sample_size = self.trajectory["observations"].shape[0]
        mbsize = sample_size // self.num_minibatches
        start = self.minibatch_count * mbsize
        indices = np.arange(start, min(start + mbsize, sample_size))
        minibatch = {key: val[indices] for key, val in self.trajectory.items()
                     if isinstance(val, np.ndarray)}

        self.minibatch_count += 1
        if self.minibatch_count == self.num_minibatches:
            self.minibatch_count = 0
            self.epoch_count += 1
            if self.shuffle_before_epoch and not self.trajectory_is_stale():
                self.shuffle_trajectory()

        for transform in self.transforms:
            transform(minibatch)
        return minibatch


class SavedRewardsResetsRunner(BaseRunner1):
    """ Saves rewards and resets to an internall collection. """

    def __init__(self, runner):
        if isinstance(runner, TrajectorySampler):
            unwrapped_runner = runner.runner
        else:
            unwrapped_runner = runner
        assert isinstance(unwrapped_runner, EnvRunner1)
        super().__init__(runner.env, runner.policy, step_var=runner.step_var)
        self.runner = runner

        self.rewards = []
        self.resets = []
        true_get_next = unwrapped_runner.get_next

        def wrapped_get_next():
            trajectory = true_get_next()
            self.rewards.append(trajectory["rewards"])
            self.resets.append(trajectory["resets"])
            return trajectory

        unwrapped_runner.get_next = wrapped_get_next

    def get_next(self):
        return self.runner.get_next()

    def get_rewards_resets(self):
        """ Returns tuple of (rewards, resets) """

        return np.concatenate(self.rewards, 0), np.concatenate(self.resets, 0)

    def clear_rewards_resets(self):
        """ Clears underlying collections of rewards and resets. """
        self.rewards.clear()
        self.resets.clear()

    def __getattr__(self, attr):
        return getattr(self.runner, attr)


class MergeTimeBatch:
    """ Merges first two axes typically representing time and env batch. """

    def __call__(self, trajectory):
        assert trajectory["resets"].ndim == 2, trajectory["resets"].shape
        for key, val in filter(lambda kv: isinstance(kv[1], np.ndarray),
                               trajectory.items()):
            trajectory[key] = np.reshape(val, (-1, *val.shape[2:]))


class NormalizeAdvantages:
    """ Normalizes advantages. """

    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def __call__(self, trajectory):
        advantages = trajectory["advantages"]
        trajectory["advantages"] = ((advantages - advantages.mean())
                                    / (advantages.std() + self.epsilon))


def make_ppo_runner(env, policy, num_runner_steps, gamma=0.99, lambda_=0.95,
                    num_epochs=3, num_minibatches=4):
    """ Returns env runner for PPO """
    transforms = [GAE(policy, gamma=gamma, lambda_=lambda_, normalize=False)]
    if not policy.is_recurrent() and getattr(env.unwrapped, "nenvs", None):
        transforms.append(MergeTimeBatch())
    runner = EnvRunner1(env, policy, num_runner_steps, transforms=transforms)
    runner = TrajectorySampler(runner, num_epochs=num_epochs,
                               num_minibatches=num_minibatches,
                               transforms=[NormalizeAdvantages()])
    return runner
