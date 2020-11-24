import numpy as np
import torch
from torch.nn import Module

from selection.spaces import SkipSpace, ChooseSpace


class ControllerModel(Module):
    """ Simple controller model. """

    def __init__(self, space, device, hidden_size=64, init=None):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.h_x, self.c_x = self.init_hidden()
        self.space = space
        self.start_embedding = torch.autograd.Variable(torch.ones(1, 1, hidden_size))
        self.embedding = torch.nn.Embedding(self.space.maxchoices, hidden_size)
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size=hidden_size, batch_first=True)  # init
        self.logits = torch.nn.Linear(hidden_size, self.space.maxchoices)  # init

        # Attention mechanism over previous branches is used for skip connections.
        self.key = torch.nn.Linear(hidden_size, hidden_size)  # init
        self.query = torch.nn.Linear(hidden_size, hidden_size)  # init
        self.connection_logits = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 2),  # THIS IS NEW
            #             torch.nn.Tanh(),
        )

    def single_step(self, n, logits, samples, lstm_outputs=None, keys=None,
                    lstm_inputs=None):
        """ Performs a single step and updates logits and samples. """
        if lstm_inputs is None:
            lstm_inputs = (self.start_embedding if not logits
                           else self.embedding(samples[len(logits) - 1]))
        lstm, _ = self.lstm(lstm_inputs.to(self.device), (self.h_x, self.c_x))  # lstm_inputs.cuda()
        lstm = lstm[:, -1]

        if lstm_outputs is not None:
            lstm_outputs.append(lstm)
        newlogits = self.logits(lstm)[:, :n]
        if isinstance(samples, list):
            proba_distr = torch.nn.functional.log_softmax(newlogits, dim=-1).data
            proba_distr = proba_distr.exp()
            samples.append(torch.multinomial(proba_distr, num_samples=1))
        newlogits = torch.nn.functional.pad(newlogits, [0, self.space.maxchoices - n, 0, 0], value=0.)
        logits.append(newlogits)
        if keys is not None:
            keys.append(self.key(lstm))
            # print(keys)

    def step_connections(self, lstm_outputs, logits, samples, keys):
        """ Predicts wether the model should have skip connections. """
        lstm = lstm_outputs[-1]
        lstm_outputs = lstm_outputs[:-1]
        keys = keys[:-1]
        query = torch.tanh(torch.stack(keys, 1) + self.query(lstm)[:, None])
        connection_logits = self.connection_logits(query)
        # pylint: disable=invalid-unary-operand-type

        proba_distr = torch.nn.functional.log_softmax(connection_logits, dim=-1).data
        proba_distr = proba_distr.exp()

        connections = torch.multinomial(torch.reshape(proba_distr, (-1, 2)), num_samples=1)
        connections = torch.reshape(connections, [keys[0].shape[0], len(keys)])

        connection_logits = torch.nn.functional.pad(connection_logits, [0, self.space.maxchoices - 2],
                                                    value=0.)
        logits.extend(connection_logits[:, i] for i in range(len(lstm_outputs)))
        if isinstance(samples, list):
            samples.extend(connections[:, i][..., None] for i in range(len(lstm_outputs)))

        connections = connections.float()  # type(torch.FloatTensor)
        
        # connections.unsqueeze(-2):     [? x 1 x len(keys)]
        # torch.stack(lstm_outputs, 1):  [B x len(keys) x n_channels]
        lstm_inputs = torch.matmul(connections.unsqueeze(-2),
                                   torch.stack(lstm_outputs, 1))
        lstm_inputs /= (1. + torch.sum(connections))
        return lstm_inputs

    def forward(self, samples=None):  # pylint: disable=arguments-differ
        logits = []
        if samples is None:
            samples = []
        else:
            if isinstance(samples, np.ndarray):
                samples = torch.from_numpy(samples).to(self.device)
            if samples.shape[0] != 1 or samples.ndim != 2:
                raise ValueError(f"samples have invalid shape: {samples.shape}; "
                                 "expected (1, None)")
            samples = torch.transpose(samples, dim0=1, dim1=0)[..., None]  # ???
        lstm_outputs = []
        keys = []

        lstm_inputs = None
        for j, space in enumerate(self.space.spaces):
            if isinstance(space, SkipSpace):
                lstm_inputs = self.step_connections(lstm_outputs, logits, samples, keys)
                continue

            elif isinstance(space, ChooseSpace):
                self.single_step(len(space.spaces), logits, samples, lstm_inputs=lstm_inputs)
                space = space.spaces[int(torch.squeeze(samples[len(logits) - 1]))]
                lstm_inputs = None

            for i, n in enumerate(space.ns):
                self.single_step(n, logits, samples,
                                 lstm_inputs=lstm_inputs,
                                 lstm_outputs=(None if i < len(space.ns) - 1
                                               else lstm_outputs),
                                 keys=None if i < len(space.ns) - 1 else keys)

            lstm_inputs = None

        # logits: [B x len(keys) x maxchoices]
        logits = torch.stack(logits, 1)  # 1 x len x maxchoices
        samples = (torch.cat(samples, -1) if isinstance(samples, list) else torch.transpose(torch.squeeze(samples, -1), dim0=1, dim1=0))
        self.init_hidden()
        return logits, samples

    def init_hidden(self):
        hidden = torch.autograd.Variable(torch.zeros((1, 1, self.hidden_size), requires_grad=True)).to(self.device)
        cell = torch.autograd.Variable(torch.zeros((1, 1, self.hidden_size))).to(self.device)
        return (hidden, cell)
