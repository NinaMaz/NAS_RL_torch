from builtins import print
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
from utils.arguments import args
from env.envs import make
from runners.runners import EnvRunnerNoController
from utils.nas_summary import summary
from utils.trajectory_transforms import get_nas_transforms

from learners.learners import PPOLearner, ScoredLearner
from selection.base_layers import selectable_nature_dqn_base_layers
from selection.select_layers import SelectLayer, SelectModelFromLayers
from train_nas_norunner import make_nas_env
from utils.additional import get_device, arr2str


class ShuffleNetOneShot(nn.Module):

    def __init__(self, layers, input_size=84, architecture=None):
        super(ShuffleNetOneShot, self).__init__()

        # change it if you change gym env
        assert input_size % 84 == 0
        self.architecture = architecture

        first_layers, middle_layers, last_layers = layers

        # building first layer
        self.first_block = nn.Sequential(*first_layers)

        # building the intermediate layers
        # Arch space = num_l  * (num_c_0 ** num_c_1 ** num_c_3 **... num_c_n); such that num_c_i > num_c_j for every i>j
        # Arch len = summation_over_i{ param_i }

        self.arch_len = 0
        self.arch_lens = [0]
        self.features = torch.nn.ModuleList()
        for layer_i in middle_layers:
            if isinstance(layer_i, SelectLayer):
                layer_i_params_keys = layer_i.choices.keys()
                self.arch_lens.append(len(layer_i_params_keys))
                self.features.append(torch.nn.ModuleList())
                for param in layer_i_params_keys:
                    self.arch_len += 1
                    self.features[-1].append(layer_i)
        self.cumulative_arch_index = np.cumsum(self.arch_lens)

        # building last Layer
        self.last_block = nn.Sequential(*last_layers)

        self._initialize_weights()

    def forward(self, x, architecture=None):
        assert architecture or self.architecture, 'architecture must be feed to the super net.'
        architecture = self.architecture if architecture is None else architecture
        assert self.arch_len == len(architecture), 'Architecture: %s is mismatch with the archLen:%d' % (
            ','.join(arr2str(architecture)), self.arch_len)

        # forward the first layer
        x = self.first_block(x)

        # forward the intermediate layers
        for i, arch in enumerate(self.features):
            selected = architecture[self.cumulative_arch_index[i]: self.cumulative_arch_index[i + 1]]
            # print('selected >>', selected)
            # print('i >>', i)
            print('arc_len >>', len(arch))
            if i < len(arch):
                if isinstance(arch[i], SelectLayer):
                    x = arch[i](x, selected)
                else:
                    x = arch[i](x)

        # forward the last layers
        x = self.last_block(x)
        return x

    def _initialize_weights(self):
        # TODO: initialize model_instance weights correctly
        pass


def make_rl_learner(env, args, layers, device):
    """ Creates and returns scored reinforcement learning learner. """
    first_layers, block_layers, last_layers = layers
    base_layer = first_layers + block_layers + last_layers

    model = SelectModelFromLayers([env.action_space.n, 1], base_layers=base_layer, skip_connections=False)
    runner = PPOLearner.make_runner(env, args, model, device)
    ppo = PPOLearner.make_alg(runner, args, device)
    return ScoredLearner(runner, ppo)


class Value2Loss(torch.nn.Module):
    def __init__(self):
        super(Value2Loss, self).__init__()

    def forward(self, loss):
        return torch.tensor(loss, requires_grad=True)


def training_step_spos_model(model_instance, device, cand, data, loss_value, optimizer, scheduler):
    model_instance.train()
    model_instance(data, cand)
    loss_value_function = Value2Loss()
    loss_grad_value = loss_value_function(loss_value).to(device)

    optimizer.zero_grad()
    loss_grad_value.backward()

    for p in model_instance.parameters():
        if p.grad is not None and p.grad.sum() == 0:
            p.grad = None

    optimizer.step()
    scheduler.step()


if __name__ == "__main__":
    args = args['atari']
    env_id = 'FreewayNoFrameskip-v0'
    device = get_device()
    layers = selectable_nature_dqn_base_layers()

    env = make(env_id=env_id, nenvs=args['nenvs'])
    learner = make_rl_learner(env, args, layers, device)

    nas_env = make_nas_env(learner, args, device)
    nas_runner = EnvRunnerNoController(nas_env,
                                       args['num_nas_runner_steps'],
                                       asarray=False,
                                       transforms=get_nas_transforms(),
                                       step_var=learner.runner.step_var)

    nas_runner.state["env_steps"] = nas_runner.nsteps

    # Choose random action (architecture) from sample space
    space = learner.model.space

    architecture = space.sample()
    act = {'actions': architecture}
    done = nas_runner.get_next(act, [], [], [], defaultdict(list, {"actions": []}))

    print('Architecture length:', len(act['actions']))
    print('Architecture:', act['actions'])

    model = ShuffleNetOneShot(layers, architecture=architecture).to(device)

    test_data = torch.from_numpy(learner.current_data['observations']).float().to(device)
    test_outputs = model(test_data, act['actions'])

    total_params, trainable_params = summary(model, test_data.size()[1:], arch=act['actions'], device=device)

    print('Input size:', test_data.size())
    print('Output size:', test_outputs.size())
    print('Total Params:', total_params.item())
    print('Trainable Params:', trainable_params.item())
