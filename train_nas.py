import torch
from tqdm import tqdm

from controller.controller_model import ControllerModel
from env.envs import NASEnv
from env.envs import make
from learners.learners import ScoredLearner, PPOLearner
from learners.policies_algs import ControllerPolicy
from learners.policies_algs import REINFORCE
from runners.runners import EnvRunner1
from selection.base_layers import small_base_layers
from selection.select_layers import SelectModelFromLayers, reset_parameters
from utils.additional import get_device
from utils.summarizer import Summarize
from utils.trajectory_transforms import get_nas_transforms
from utils.arguments import args


def make_rl_learner(env, args, device):
    """ Creates and returns scored rienforcement learning learner. """
    model = SelectModelFromLayers([env.action_space.n, 1], base_layers=small_base_layers())

    runner = PPOLearner.make_runner(env, args, model, device)
    ppo = PPOLearner.make_alg(runner, args, device)
    return ScoredLearner(runner, ppo)


def make_nas_env(learner, args, device):
    """ Creates and returns neural architecture search environment. """
    env = NASEnv(learner, nsteps=args['num_learner_steps'], device=device)
    return env


def make_controller(space, device, checkpoint=None):
    """ Creates and returns controller policy. """
    model = ControllerModel(space, device).to(device)
    # if checkpoint is not None:
    # model.load_weights(checkpoint)
    model.apply(reset_parameters)
    return ControllerPolicy(model)


def train(device, args, env_id='FreewayNoFrameskip-v0', logdir=' '):
    args = args['atari']
    env = make(env_id=env_id, nenvs=args['nenvs'])
    learner = make_rl_learner(env, args, device)
    nasenv = make_nas_env(learner, args, device)
    controller = make_controller(learner.model.space, device)
    nasrunner = EnvRunner1(nasenv, controller, args['num_runner_steps'],
                           asarray=False,
                           transforms=get_nas_transforms(),
                           step_var=nasenv.summarizer.step_var)
    optimizer = torch.optim.Adam(controller.model.parameters(), args['nas_lr'], eps=args['optimizer_epsilon'])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args['num_train_steps'])

    nasalgo = REINFORCE(controller, optimizer, lr_scheduler=lr_scheduler,
                        entropy_coef=args['nas_entropy_coef'],
                        baseline_momentum=args['nas_baseline_momentum'],
                        step_var=nasenv.summarizer.step)
    with tqdm(total=args['num_train_steps']) as pbar:
        while int(learner.runner.step_var) < args['num_train_steps']:
            pbar.update(int(learner.runner.step_var) - pbar.n)
            trajectory = nasrunner.get_next()
            nasalgo.step(trajectory)
    torch.save(learner.model.state_dict(), logdir + 'nas_learnt_sample.pth')
    torch.save(controller.model.state_dict(), logdir + 'nas_controller.pth')
    return controller, learner


if __name__ == "__main__":
    device = get_device()
    args = args
    _, _ = train(device, args)