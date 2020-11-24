from collections import deque

import gym
import gym.spaces as spaces
import numpy as np
import torch
from atari_py import list_games
from gym import Env
from gym.spaces import Box

from selection.select_layers import instantiate_model
from utils.summarizer import Summarize

from utils.wrappers import (
    EpisodicLife,
    FireReset,
    StartWithRandomActions,
    MaxBetweenFrames,
    SkipFrames,
    ImagePreprocessing,
    ClipReward
)
from .env_batch import ParallelEnvBatch


class QueueFrames(gym.ObservationWrapper):
    """ Queues specified number of frames together. """

    def __init__(self, env, nframes=4, concat=False):
        super(QueueFrames, self).__init__(env)
        self.obs_queue = deque([], maxlen=nframes)
        self.concat = concat
        ospace = self.observation_space
        if self.concat:
            oshape = ospace.shape[:-1] + (ospace.shape[-1] * nframes,)
        else:
            oshape = ospace.shape + (nframes,)
        if len(oshape) == 4:
            oshape_new = (oshape[3], oshape[2], oshape[0], oshape[1])
        else:
            oshape_new = (oshape[2], oshape[0], oshape[1])
        self.observation_space = spaces.Box(ospace.low.min(), ospace.high.max(),
                                            oshape_new, ospace.dtype)

    def observation(self, observation):

        self.obs_queue.append(observation)
        if self.concat:
            outputs = np.concatenate(self.obs_queue, -1)
        else:
            outputs = np.stack(self.obs_queue, -1)
        if len(outputs.shape) == 4:
            return outputs.transpose((3, 2, 0, 1))
        else:
            return outputs.transpose((2, 0, 1))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.obs_queue.maxlen - 1):
            self.obs_queue.append(obs)
        return self.observation(obs)


def list_envs(env_type):
    """ Returns list of envs ids of given type. """
    ids = {
        "atari": list("".join(c.capitalize() for c in g.split("_"))
                      for g in list_games()),
        "mujoco": [
            "Reacher",
            "Pusher",
            "Thrower",
            "Striker",
            "InvertedPendulum",
            "InvertedDoublePendulum",
            "HalfCheetah",
            "Hopper",
            "Swimmer",
            "Walker2d",
            "Ant",
            "Humanoid",
            "HumanoidStandup",
        ]
    }
    return ids[env_type]


def is_atari_id(env_id):
    """ Returns True if env_id corresponds to an Atari env. """
    env_id = env_id[:env_id.rfind("-")]
    for postfix in ("Deterministic", "NoFrameskip"):
        if env_id.endswith(postfix):
            env_id = env_id[:-len(postfix)]

    atari_envs = set(list_envs("atari"))
    return env_id in atari_envs


def is_mujoco_id(env_id):
    """ Returns True if env_id corresponds to MuJoCo env. """
    env_id = "".join(env_id.split("-")[:-1])
    mujoco_ids = set(list_envs("mujoco"))
    return env_id in mujoco_ids


def get_seed(nenvs=None, seed=None):
    """ Returns seed(s) for specified number of envs. """
    if nenvs is None and seed is not None and not isinstance(seed, int):
        raise ValueError("when nenvs is None seed must be None or an int, "
                         f"got type {type(seed)}")
    if nenvs is None:
        return seed or 0
    if isinstance(seed, (list, tuple)):
        if len(seed) != nenvs:
            raise ValueError(f"seed must have length {nenvs} but has {len(seed)}")
        return seed
    if seed is None:
        seed = list(range(nenvs))
    elif isinstance(seed, int):
        seed = [seed] * nenvs
    else:
        raise ValueError(f"invalid seed: {seed}")
    return seed


def nature_dqn_env(env_id, nenvs=None, summarize=True, episodic_life=True, clip_reward=True):
    """ Wraps env as in Nature DQN paper. """
    assert is_atari_id(env_id)
    if "NoFrameskip" not in env_id:
        raise ValueError(f"env_id must have 'NoFrameskip' but is {env_id}")
    seed = get_seed(nenvs)
    if nenvs is not None:
        env = ParallelEnvBatch([
            lambda i=i, s=s: nature_dqn_env(env_id, summarize=False, episodic_life=episodic_life, clip_reward=False)
            for i, s in enumerate(seed)
        ])
        if summarize:
            env = Summarize.reward_summarizer(env, prefix=env_id)
        if clip_reward:
            env = ClipReward(env)
        return env

    env = gym.make(env_id)
    env.seed(seed)
    return nature_dqn_wrap(env,
                           summarize=summarize,
                           episodic_life=episodic_life,
                           clip_reward=clip_reward)


def nature_dqn_wrap(env, summarize=True, episodic_life=True, clip_reward=True):
    """ Wraps given env as in nature DQN paper. """
    if episodic_life:
        env = EpisodicLife(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireReset(env)
    env = StartWithRandomActions(env, max_random_actions=30)
    env = MaxBetweenFrames(env)
    env = SkipFrames(env, 4)
    env = ImagePreprocessing(env, width=84, height=84, grayscale=True)
    env = QueueFrames(env, 4)
    if clip_reward:
        env = ClipReward(env)
    return env


def mujoco_env(env_id, nenvs=None, seed=None, summarize=True,
               normalize_obs=True, normalize_ret=True):
    """ Creates and wraps MuJoCo env. """
    assert is_mujoco_id(env_id)
    seed = get_seed(nenvs, seed)
    if nenvs is not None:
        env = ParallelEnvBatch([
            lambda s=s: mujoco_env(env_id, seed=s, summarize=False,
                                   normalize_obs=False, normalize_ret=False)
            for s in seed])
        return mujoco_wrap(env, summarize=summarize, normalize_obs=normalize_obs,
                           normalize_ret=normalize_ret)

    env = gym.make(env_id)
    env.seed(seed)
    return mujoco_wrap(env, summarize=summarize, normalize_obs=normalize_obs,
                       normalize_ret=normalize_ret)


def mujoco_wrap(env, summarize=True, normalize_obs=True, normalize_ret=True):
    """ Wraps given env as a mujoco env. """
    if summarize:
        env = Summarize.reward_summarizer(env)
    if normalize_obs or normalize_ret:
        env = Normalize(env, obs=normalize_obs, ret=normalize_ret)
    return env


def make(env_id, nenvs=None, seed=None, **kwargs):
    """ Creates env with standard wrappers. """
    if is_atari_id(env_id):
        return nature_dqn_env(env_id, nenvs, **kwargs)
    if is_mujoco_id(env_id):
        return mujoco_env(env_id, nenvs, seed=seed, **kwargs)

    def _make(seed):
        env = gym.make(env_id, **kwargs)
        env.seed(seed)
        return env

    seed = get_seed(nenvs, seed)
    if nenvs is None:
        return _make(seed)
    return ParallelEnvBatch([lambda s=s: _make(s) for s in seed])


class NASEnv(Env):
    """ Neural Architecture Search environment. """

    def __init__(self, learner, nsteps, device, logdir=None, log_period=1):
        self.learner = learner
        self.nsteps = nsteps
        self.logdir = logdir
        self.log_period = log_period
        self.observation_space = Box(low=1, high=1, shape=(), dtype=np.int32)
        self.action_space = learner.select_model.space
        self.device = device

    def step(self, action):
        self.learner.select_model.select(action)
        self.learner.select_model.to(self.device)

        env = self.learner.runner.env
        x = torch.rand(1, *env.observation_space.shape).to(self.device)
        instantiate_model(self.learner.select_model.to(self.device), x)

        for _ in range(self.nsteps):
            self.learner.learn(int(self.learner.runner.step_var) + 1, self.logdir, self.log_period)

        rew = self.learner.get_score()
        return None, rew, True, {}

    def reset(self):
        return None

    def render(self, mode="human"):
        raise NotImplementedError()
