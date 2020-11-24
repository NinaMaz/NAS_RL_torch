import numpy as np
import torch

from base.base import Learner
from runners.runners import make_ppo_runner, SavedRewardsResetsRunner
from selection.select_layers import SelectModelFromLayers
from utils.additional import GPU_ids
from .policies_algs import ActorCriticPolicy, PPO


class PPOLearner(Learner):
    """ Proximal Policy Optimization learner. """

    @staticmethod
    def get_defaults(env_type="atari"):
        defaults = {
            "atari": {
                "num_train_steps": 10e6,
                "nenvs": 8,
                "num_runner_steps": 128,
                "gamma": 0.99,
                "lambda_": 0.95,
                "num_epochs": 3,
                "num_minibatches": 4,
                "cliprange": 0.1,
                "value_loss_coef": 0.25,
                "entropy_coef": 0.01,
                "max_grad_norm": 0.5,
                "lr": 2.5e-4,
                "optimizer_epsilon": 1e-5,
            },
            "mujoco": {
                "num_train_steps": 1e6,
                "nenvs": dict(type=int, default=None),
                "num_runner_steps": 2048,
                "gamma": 0.99,
                "lambda_": 0.95,
                "num_epochs": 10,
                "num_minibatches": 32,
                "cliprange": 0.2,
                "value_loss_coef": 0.25,
                "entropy_coef": 0.,
                "max_grad_norm": 0.5,
                "lr": 3e-4,
                "optimizer_epsilon": 1e-5,
            }
        }
        return defaults.get(env_type)

    @staticmethod
    def make_runner(env, args, model, device):

        policy = ActorCriticPolicy(model, device)
        kwargs = args  # vars(args)
        runner_kwargs = {key: kwargs[key] for key in
                         ["gamma", "lambda_", "num_epochs", "num_minibatches"]
                         if key in kwargs}
        runner = make_ppo_runner(env, policy, args['num_runner_steps'],
                                 **runner_kwargs)
        return runner

    @staticmethod
    def make_alg(runner, args, device):
        lr = args['lr']
        model = runner.policy.model
        model.to(device)
        model = torch.nn.DataParallel(model, device_ids=GPU_ids)

        if "optimizer_epsilon" in args:
            optimizer = torch.optim.Adam(model.parameters(), lr, eps=args['optimizer_epsilon'])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args['num_train_steps'])

        kwargs = args
        ppo_kwargs = {key: kwargs[key]
                      for key in ["value_loss_coef", "entropy_coef",
                                  "cliprange", "max_grad_norm"]
                      if key in kwargs}

        ppo = PPO(runner.policy, device, optimizer, lr_scheduler, **ppo_kwargs)
        return ppo

    def learning_body(self):
        # self.runner.step_var+=1
        data = self.runner.get_next()
        loss = self.alg.step(data)
        # save_to_file('new_logs/random_loss.csv', {'loss':loss})
        yield data, loss
        while not self.runner.trajectory_is_stale():
            data = self.runner.get_next()
            loss = self.alg.step(data)
            yield data, loss


class ScoredLearner(Learner):
    """ Scored learner. """

    # pylint: disable=abstract-method
    def __init__(self, runner, alg):
        if not isinstance(alg.model, SelectModelFromLayers):
            raise ValueError("alg.model must be an instance of SelectModel, "
                             f"got type {type(alg.model)} instead")
        runner = SavedRewardsResetsRunner(runner)
        super().__init__(runner=runner, alg=alg)
        self.select_model = alg.model

        self.current_data = None
        self.current_loss = None

    def get_score(self):
        """ Returns score over the last learning trajectory. """
        rewards, resets = self.runner.get_rewards_resets()
        self.runner.clear_rewards_resets()
        assert rewards.ndim == 1 and resets.ndim == 1, (rewards.ndim, resets.ndim)
        assert rewards.shape[0] == resets.shape[0], (rewards.shape, resets.shape)
        scores = [0]
        for t in reversed(range(rewards.shape[0])):
            if resets[t]:
                scores.append(0)
            scores[-1] += rewards[t]
        return np.mean(scores)

    def learning_body(self):
        data = self.runner.get_next()
        loss = self.alg.step(data)
        self.current_data = data
        self.current_loss = loss
        yield data, loss
        while not self.runner.trajectory_is_stale():
            data = self.runner.get_next()
            loss = self.alg.step(data)
            yield data, loss
