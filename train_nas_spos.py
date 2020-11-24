from collections import defaultdict

import numpy as np
import random
import torch
from tqdm import tqdm

from env.envs import make
from runners.runners import EnvRunnerNoController
from selection.base_layers import selectable_nature_dqn_base_layers
from spos.supernet import ShuffleNetOneShot, make_rl_learner, training_step_spos_model
from train_nas_norunner import make_nas_env
from utils.additional import get_device, save_checkpoint, arr2str, GPU_ids
from utils.additional import get_parameters, storage_saver
from utils.arguments import args
from utils.trajectory_transforms import get_nas_transforms

# conda activate py36 && cd /home/samir/Desktop/repositories/my_new_RL_NAS/ENAS_RL_torch && nohup python train_nas_spos.py &

args = args['atari']
env_id = 'FreewayNoFrameskip-v0'
device = get_device()
layers = selectable_nature_dqn_base_layers()

if __name__ == '__main__':

    storage_saver.set_architecture([None] * 2)
    env = make(env_id=env_id, nenvs=args['nenvs'])
    learner = make_rl_learner(env, args, layers, device)
    nas_env = make_nas_env(learner, args, device)
    nas_runner = EnvRunnerNoController(nas_env,
                                       nsteps=args['num_nas_runner_steps'],
                                       asarray=False,
                                       transforms=get_nas_transforms(),
                                       step_var=learner.runner.step_var)

    model = ShuffleNetOneShot(layers)
    model = torch.nn.DataParallel(model, device_ids=GPU_ids)
    model = model.to(device)

    optimizer = torch.optim.SGD(get_parameters(model),
                                lr=args['spos_learning_rate'],
                                weight_decay=args['spos_weight_decay'])

    step_fun = lambda step: (1.0 - step / args['num_train_steps']) if step <= args['num_train_steps'] else 0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  step_fun,
                                                  last_epoch=-1)

    trajectory = defaultdict(list, {"actions": []})
    observations = []
    rewards = []
    resets = []
    loss = np.inf

    model.train()

    with tqdm(total=args['num_train_steps']) as pbar:
        while int(learner.runner.step_var) < args['num_train_steps']:
            pbar.update(int(learner.runner.step_var) - pbar.n)

            trajectory = defaultdict(list, {"actions": []})
            observations = []
            rewards = []
            resets = []

            nas_runner.state["env_steps"] = nas_runner.nsteps

            for i in range(nas_runner.nsteps):

                space = learner.model.space
                print(space)
                architecture = space.sample()
                print(architecture)
                act = {'actions': architecture}

                done = nas_runner.get_next(act, observations, rewards, resets, trajectory)

                data = torch.from_numpy(learner.current_data['observations']).float().to(device)
                loss = learner.current_loss

                if not nas_runner.nenvs and np.all(done):
                    nas_runner.state["env_steps"] = i + 1
                    nas_runner.state["latest_observation"] = nas_runner.env.reset()

                training_step_spos_model(model, device, architecture, data, loss, optimizer, scheduler)

            trajectory.update(observations=observations, rewards=rewards, resets=resets)
            if nas_runner.asarray:
                for key, val in trajectory.items():
                    try:
                        trajectory[key] = np.asarray(val)
                    except ValueError:
                        raise ValueError(f"cannot convert value under key '{key}' to np.ndarray")

            trajectory["state"] = nas_runner.state

            for transform in nas_runner.transforms:
                transform(trajectory)

            save_checkpoint(model.state_dict(), iters=int(learner.runner.step_var), tag=arr2str(architecture))
