from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from env.envs import make
from learners.policies_algs import REINFORCE
from runners.runners import EnvRunnerNoController
from train_nas import make_rl_learner, make_nas_env, make_controller
from utils.additional import get_device
from utils.arguments import args
from utils.trajectory_transforms import get_nas_transforms


def train(device, args, env_id='FreewayNoFrameskip-v0', logdir=' '):
    args = args['atari']
    env = make(env_id=env_id, nenvs=args['nenvs'])
    learner = make_rl_learner(env, args, device)
    nasenv = make_nas_env(learner, args, device)
    controller = make_controller(learner.model.space, device)
    nasrunner = EnvRunnerNoController(nasenv, args['num_nas_runner_steps'],
                                      asarray=False,
                                      transforms=get_nas_transforms(),
                                      step_var=nasenv.summarizer.step_var)
    optimizer = torch.optim.Adam(controller.model.parameters(), args['nas_lr'], eps=args['optimizer_epsilon'])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args['num_train_steps'])

    nasalgo = REINFORCE(controller, optimizer, lr_scheduler=lr_scheduler,
                        entropy_coef=args['nas_entropy_coef'],
                        baseline_momentum=args['nas_baseline_momentum'],
                        step_var=nasenv.summarizer.step_var)

    with tqdm(total=args['num_train_steps']) as pbar:
        while int(learner.runner.step_var) < args['num_train_steps']:
            pbar.update(int(learner.runner.step_var) - pbar.n)

            trajectory = defaultdict(list, {"actions": []})
            observations = []
            rewards = []
            resets = []
            if controller.is_recurrent():
                nasrunner.state["policy_state"] = controller.get_state()
            nasrunner.state["env_steps"] = nasrunner.nsteps

            for i in range(nasrunner.nsteps):
                act = controller.act(nasrunner.state["latest_observation"])
                done = nasrunner.get_next(act, observations, rewards, resets, trajectory)
                # Only reset if the env is not batched. Batched envs should auto-reset.

                if not nasrunner.nenvs and np.all(done):
                    nasrunner.state["env_steps"] = i + 1
                    nasrunner.state["latest_observation"] = nasrunner.env.reset()
                    if nasrunner.cutoff or (nasrunner.cutoff is None and controller.is_recurrent()):
                        pass

            trajectory.update(observations=observations, rewards=rewards, resets=resets)
            if nasrunner.asarray:
                for key, val in trajectory.items():
                    try:
                        trajectory[key] = np.asarray(val)
                    except ValueError:
                        raise ValueError(f"cannot convert value under key '{key}' to np.ndarray")
            trajectory["state"] = nasrunner.state

            for transform in nasrunner.transforms:
                transform(trajectory)

            nasalgo.step(trajectory)
    torch.save(learner.model.state_dict(), logdir + 'nas_learnt_sample.pth')
    torch.save(controller.model.state_dict(), logdir + 'nas_controller.pth')
    return controller, learner


if __name__ == "__main__":
    device = get_device()
    args = args
    _, _ = train(device, args)
