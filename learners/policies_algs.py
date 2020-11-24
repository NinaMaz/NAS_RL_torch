import time

import numpy as np
import torch

from base.base import Policy, BaseAlgorithm
from controller.controller_model import ControllerModel
from utils.additional import storage_saver


class ActorCriticPolicy(Policy):
    """ Actor critic policy with discrete number of actions. """

    def __init__(self, model, device, distribution=None):
        self.model = model
        self.distribution = distribution
        self.device = device

    def act(self, inputs, state=None, update_state=True, training=False):
        if state is not None:
            raise NotImplementedError()
        if training:
            observations = torch.tensor(inputs["observations"])
        else:
            observations = torch.tensor(inputs)
        observations = observations.type(torch.FloatTensor).to(self.device)
        expand_dims = 4 - observations.ndim  # 4
        observations = observations[(None,) * expand_dims]

        *distribution_inputs, values = self.model(observations)
        squeeze_dims = tuple(range(expand_dims))
        if squeeze_dims:
            distribution_inputs = [torch.squeeze(inputs.to(self.device), squeeze_dims[0]) for inputs in
                                   distribution_inputs]  # 0
            values = torch.squeeze(values.to(self.device), squeeze_dims[0])  # 0
        if self.distribution is None:
            if len(distribution_inputs) == 1:
                distribution = torch.distributions.Categorical(logits=torch.cat(distribution_inputs, dim=0))
            elif len(distribution_inputs) == 2:
                distribution = torch.distributions.multivariate_normal.MultivariateNormal(*distribution_inputs,
                                                                                          torch.eye(
                                                                                              len(distribution_inputs)))
            else:
                raise ValueError(f"model has {len(distribution_inputs)} "
                                 "outputs to create a distribution, "
                                 "expected a single output for categorical "
                                 "and two outputs for normal distributions")
        else:
            distribution = self.distribution(*distribution_inputs)
        if training:
            return {"distribution": distribution, "values": values}

        actions = distribution.sample()

        log_prob = distribution.log_prob(actions)
        return {"actions": actions.cpu().detach().numpy(),
                "log_prob": log_prob.cpu().detach().numpy(),
                "values": values.cpu().detach().numpy()}


class PPO(BaseAlgorithm):
    """ Proximal Policy Optimization algorithm.
    See [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 policy,
                 device,
                 optimizer=None,
                 lr_scheduler=None,
                 cliprange=0.2,
                 value_loss_coef=0.25,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 step_var=None):
        super().__init__(model=policy.model, optimizer=optimizer, lr_scheduler=lr_scheduler, step_var=step_var)
        self.policy = policy
        self.cliprange = cliprange
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device  # list(policy.model.parameters())[0].device
        ts = time.gmtime()
        self.tmstmp = time.strftime("%m%d%H%M", ts)

    def policy_loss(self, trajectory, act=None):
        """ Compute policy loss (including entropy regularization). """
        if act is None:
            act = self.policy.act(trajectory, training=True).to(self.device)
        if "advantages" not in trajectory:
            raise ValueError("trajectory does not contain 'advantages'")
        old_log_prob = torch.from_numpy(trajectory["log_prob"]).to(self.device)
        advantages = torch.from_numpy(trajectory["advantages"]).to(self.device)
        actions = torch.from_numpy(trajectory["actions"]).to(self.device)
        log_prob = act["distribution"].log_prob(actions)
        if log_prob.shape != old_log_prob.shape:
            raise ValueError("trajectory has mismatched shapes: "
                             f"log_prob.shape={log_prob.shape} "
                             f"old_log_prob.shape={old_log_prob.shape}")
        if log_prob.shape != advantages.shape:
            raise ValueError("trajectory has mismatched shapes: "
                             f"log_prob.shape={log_prob.shape} "
                             f"advantages.shape={advantages.shape}")

        ratio = torch.exp(log_prob - old_log_prob)
        policy_loss = -ratio * advantages
        if self.cliprange is not None:
            ratio_clipped = torch.clamp(ratio, 1. - self.cliprange, 1. + self.cliprange)
            policy_loss_clipped = -ratio_clipped * advantages
            policy_loss = torch.max(policy_loss, policy_loss_clipped)

        policy_loss = torch.mean(policy_loss)
        entropy = torch.mean(act["distribution"].entropy())
        return policy_loss - self.entropy_coef * entropy

    def value_loss(self, trajectory, act=None):
        """ Computes value loss. """
        if act is None:
            act = self.policy.act(trajectory, training=True).to(self.device)
        if "value_targets" not in trajectory:
            raise ValueError("trajectory does not contain 'value_targets'")

        value_targets = torch.from_numpy(trajectory["value_targets"]).to(self.device)
        old_value_preds = torch.from_numpy(trajectory["values"]).to(self.device)
        values = act["values"]

        if values.shape != value_targets.shape:
            raise ValueError("trajectory has mismatched shapes "
                             f"values.shape={values.shape} "
                             f"value_targets.shape={value_targets.shape}")

        value_loss = torch.pow(values - value_targets, 2)
        if self.cliprange is not None:
            values_clipped = old_value_preds + torch.clamp(values - old_value_preds, -self.cliprange, self.cliprange)
            value_loss_clipped = torch.pow(values_clipped - value_targets, 2)
            value_loss = torch.max(value_loss, value_loss_clipped)

        value_loss = torch.mean(value_loss)
        return value_loss

    def loss(self, data):
        """ Returns ppo loss for given data (trajectory dict). """
        act = self.policy.act(data, training=True)
        policy_loss = self.policy_loss(data, act)
        value_loss = self.value_loss(data, act)
        loss = policy_loss + self.value_loss_coef * value_loss
        # save_to_file('new_logs/nature_cnn_policy_loss'+self.tmstmp+'.csv', {'p_loss':policy_loss.cpu().detach().numpy()})
        # save_to_file('new_logs/nature_cnn_value_loss'+self.tmstmp+'.csv', {'v_loss':value_loss.cpu().detach().numpy()})
        return loss

    def preprocess_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)


class ControllerPolicy(Policy):
    """ Controller policy. """

    def __init__(self, model):
        if not isinstance(model, ControllerModel):
            raise TypeError("model expected to be an instance of ControllerModel, "
                            f"got type {type(model)}")
        self.model = model

    def act(self, inputs, state=None, update_state=True, training=False):
        _ = update_state
        if state is not None:
            raise NotImplementedError()
        if training:
            samples = inputs["actions"]
            logits = [self.model(s[None])[0] for s in samples]
            maxlen = max(l.shape[1] for l in logits)
            for i, lgts in enumerate(logits):
                logits[i] = torch.nn.functional.pad(lgts, [0, 0, 0, maxlen - lgts.shape[1], 0, 0], value=0.)
            logits = torch.cat(logits, 0)  # batch_size x len x maxchoices
            distribution = torch.distributions.Categorical(logits=logits)
            return {"distribution": distribution}

        logits, samples = self.model()
        return {"actions": torch.squeeze(samples, 0).cpu().detach().numpy()}


class REINFORCE(BaseAlgorithm):
    """ REINFORCE algorithm with exponential moving average baseline. """

    def __init__(self,
                 policy,
                 optimizer=None,
                 lr_scheduler=None,
                 baseline_momentum=0.9,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 step_var=None):
        super().__init__(model=policy.model, optimizer=optimizer, lr_scheduler=lr_scheduler, step_var=step_var)

        self.policy = policy
        self.device = list(policy.model.parameters())[0].device
        self.baseline = torch.autograd.Variable(torch.zeros(1), requires_grad=False).to(self.device)
        self.baseline_momentum = baseline_momentum
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        ts = time.gmtime()
        self.tmstmp = time.strftime("%m%d%H%M", ts)

    def loss(self, data):
        if not np.all(data["resets"][-1]):
            raise ValueError(
                "reinforce requires episode to end to compute loss, "f"got resets[-1] = {data['resets'][-1]}")
        act = self.policy.act(data, training=True)

        log_prob = act["distribution"].log_prob(torch.from_numpy(data["actions"]).to(self.device))
        entropy = torch.mean(act["distribution"].entropy())

        value_targets = torch.from_numpy(data["value_targets"]).to(self.device)
        if (value_targets.ndim == log_prob.ndim + 1 and value_targets.shape[-1] == 1):
            value_targets = torch.squeeze(value_targets, -1)
        advantages = value_targets - self.baseline

        if log_prob.shape != advantages.shape:
            raise ValueError(f"log_prob.shape = {log_prob.shape} is not equal to "
                             f"advantages.shape = {advantages.shape}")
        loss = -(torch.mean(log_prob * advantages) + self.entropy_coef * entropy)
        # save_to_file('new_logs/nas_reinforce'+self.tmstmp+'.csv', {'entropy':entropy, 'loss':loss,
        #                                                                  'value_loss':torch.pow(value_targets - self.baseline, 2)})
        self.baseline *= self.baseline_momentum
        self.baseline += (1. - self.baseline_momentum) * torch.mean(value_targets)
        return loss

    def preprocess_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.max_grad_norm)
