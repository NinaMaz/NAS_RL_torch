"""Base Classes"""
from abc import ABC, abstractmethod

import torch


class Space(ABC):
    """
    Base class for 
      ---MultiDiscrete
      ---ChooseSpace
      ---SearchSpace
    Abstract space that could represent a part of the search space.
    """

    @property
    @abstractmethod
    def maxchoices(self):
        """ Returns the maximum number of classes in the search space. """

    @abstractmethod
    def __len__(self):
        """ Returns the length of the selection vector. """

    @abstractmethod
    def contains(self, element):
        """ Returns True if space contains element and False otherwise. """

    @abstractmethod
    def sample(self):
        """ Returns random sample from the space. """


class Selectable(ABC):
    """ 
    Base class for 
      --- SelectLayer
      --- SelectModel 
    The corresponding gym.space should be defined for each selectable.
    """
    space = None

    @abstractmethod
    def select(self, selection):
        """ Selects and returns part of the architecture space. """
        if not self.space.contains(selection):
            raise ValueError("selection space does not contain given selection "
                             f"{selection}")


class Learner:
    """ Base class for  """

    def __init__(self, runner, alg):
        self.runner = runner
        self.alg = alg
        self.step_var = 0

    @staticmethod
    def get_defaults(env_type="atari"):
        """ Returns default hyperparameters for specified env type. """
        return {}[env_type]

    @staticmethod
    def make_runner(env, args, model=None):
        """ Creates a runner based on the argparse Namespace. """
        raise NotImplementedError("Learner does not implement make_runner method")

    @staticmethod
    def make_alg(runner, args):
        """ Creates learner algorithm. """
        raise NotImplementedError("Learner does not implement make_alg method")

    @property
    def model(self):
        """ Model trained by the algorithm. """
        return self.alg.model

    @classmethod
    def from_env_args(cls, env, args, device, model=None):
        """ Creates a learner instance from environment and args namespace. """
        runner = cls.make_runner(env, args, model, device)
        return cls(runner, cls.make_alg(runner, args, device))

    def learning_body(self):
        """ Learning loop body. """
        data = self.runner.get_next()
        loss = self.alg.step(data)
        yield data, loss

    def learning_generator(self, nsteps, logdir=None, log_period=1):
        """ Returns learning generator object. """
        if not getattr(self.runner.step_var, "auto_update", True):
            raise ValueError("learn method is not supported when runner.step_var does not auto-update")

        # with tqdm(total=nsteps) as pbar:
        while int(self.runner.step_var) < nsteps:
            yield from self.learning_body()

    def learn(self, nsteps, logdir=None, log_period=1, save_weights=None):
        """ Performs learning for a specified number of steps. """
        if save_weights and logdir is None:
            raise ValueError("log dir cannot be None when save_weights is True")

        if save_weights is None:
            save_weights = logdir is not None
        _ = [_ for _ in self.learning_generator(nsteps, logdir, log_period)]

        if save_weights:
            torch.save(self.model.state_dict(), logdir)


class Policy(ABC):
    """ RL policy (typically wraps a keras model)."""

    def is_recurrent(self):  # pylint: disable=no-self-use
        """ Returns true if policy is recurrent. """
        return False

    def get_state(self):  # pylint: disable=no-self-use
        """ Returns current policy state. """
        return None

    def reset(self):  # pylint: disable=no-self-use
        """ Resets the state. """

    @abstractmethod
    def act(self, inputs, state=None, update_state=True, training=False):
        """ Returns `dict` of all the outputs of the policy.
        If `training=False`, then inputs can be a batch of observations
        or a `dict` containing `observations` key. Otherwise,
        `inputs` should be a trajectory dictionary with all keys
        necessary to recompute outputs for training.
        """


class BaseAlgorithm(ABC):
    """ Base algorithm. """

    def __init__(self, model, optimizer=None, lr_scheduler=None, step_var=None):
        self.model = model
        self.optimizer = optimizer or self.model.optimizer
        self.lr_scheduler = lr_scheduler

    @abstractmethod
    def loss(self, data):
        """ Computes the loss given inputs and target values. """

    def preprocess_gradients(self):
        """ Applies gradient preprocessing. """
        # pylint: disable=no-self-use
        return None

    def step(self, data):
        """ Performs single training step of the algorithm. """
        self.optimizer.zero_grad()

        loss = self.loss(data)
        loss.backward()

        self.preprocess_gradients()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return float(loss)
