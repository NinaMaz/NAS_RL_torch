import csv
import os
from datetime import datetime

import numpy as np
import pandas as pd
from gym import Wrapper

from utils.additional import write_results_csv, check_path, storage_saver, arr2str

LOG_PATH = './paper_logs/'
FILE_NAME = 'log_'
FILE_EXCITON = '.csv'


def file_is_empty(path):
    return os.stat(path).st_size == 0


def save_to_file(path, dict_saver):
    header = list(dict_saver.keys())
    values = list(dict_saver.values())
    write_results_csv(path, header, values)


class RewardSummarizer:
    """ Summarizes rewards received from environment. """

    def __init__(self, nenvs, prefix, running_mean_size=100, step_var=None):
        self.prefix = prefix
        self.step_var = step_var
        self.had_ended_episodes = np.zeros(nenvs, dtype=np.bool)
        self.rewards = np.zeros(nenvs)
        self.episode_lengths = np.zeros(nenvs)
        self.reward_queues = [[]  # deque([], maxlen=running_mean_size)
                              for _ in range(nenvs)]
        # self.reward_df = None
        self.row_list = []
        # self.dict1 = None

        self.drop_count = 0

        check_path(LOG_PATH)
        self.time_str = 'last_exp_' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
        self.file_to_save_path = ''.join([LOG_PATH, FILE_NAME, self.time_str, FILE_EXCITON])

    def should_add_summaries(self):
        """ Returns `True` if it is time to write summaries. """
        return np.all(list(self.had_ended_episodes))

    def add_summaries(self):
        """ Writes summaries. """
        dict_saver = {}
        dict_saver.update({'cand': arr2str(storage_saver.get_architecture())})
        dict_saver.update({"total_reward": np.mean(np.stack([q[-1] for q in self.reward_queues]))})
        dict_saver.update({"reward_mean": np.mean(np.stack([np.mean(q) for q in self.reward_queues]))})
        dict_saver.update({"reward_std": np.mean(np.stack([np.std(q) for q in self.reward_queues]))})
        dict_saver.update({"episode_length": np.mean(self.episode_lengths)})
        dict_saver.update({"min_reward": None})
        dict_saver.update({"max_reward": None})
        if self.had_ended_episodes.size > 1:
            dict_saver.update({"min_reward": np.max(np.stack([q[-1] for q in self.reward_queues]))})
            dict_saver.update({"max_reward": np.max(np.stack([q[-1] for q in self.reward_queues]))})

        storage_saver.set_saver_dataframe(dict_saver)

        save_to_file(self.file_to_save_path, dict_saver)

    def step(self, rewards, resets):
        """ Takes statistics from last env step and tries to add summaries.  """
        self.rewards += rewards
        self.episode_lengths[~self.had_ended_episodes] += 1
        for i, in zip(*resets.nonzero()):
            self.reward_queues[i].append(self.rewards[i])
            self.rewards[i] = 0
            self.had_ended_episodes[i] = True

        if self.should_add_summaries():
            self.add_summaries()
            self.episode_lengths.fill(0)
            self.had_ended_episodes.fill(False)

    def reset(self):
        """ Resets summarizing-related statistics. """
        self.rewards.fill(0)
        self.episode_lengths.fill(0)
        self.had_ended_episodes.fill(False)


class Summarize(Wrapper):
    """ Writes env summaries."""

    def __init__(self, env, summarizer):
        super(Summarize, self).__init__(env)
        self.summarizer = summarizer

    @classmethod
    def reward_summarizer(cls, env, prefix=None, running_mean_size=100, step_var=None):
        """ Creates an instance with reward summarizer. """
        nenvs = getattr(env.unwrapped, "nenvs", 1)
        prefix = prefix if prefix is not None else env.spec.id
        summarizer = RewardSummarizer(nenvs, prefix,
                                      running_mean_size=running_mean_size,
                                      step_var=step_var)
        return cls(env, summarizer)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        info_collection = [info] if isinstance(info, dict) else info
        done_collection = [done] if isinstance(done, bool) else done
        resets = np.asarray([info.get("real_done", done_collection[i])
                             for i, info in enumerate(info_collection)])
        self.summarizer.step(rew, resets)

        return obs, rew, done, info

    def reset(self, **kwargs):
        self.summarizer.reset()
        return self.env.reset(**kwargs)
