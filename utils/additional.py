import torch

import csv
import os
import re

MODELS_PATH = "./models_folder/"
CSV_DATA_MISMATCH = 'Row data length must match the file header length'
GPU_ids = [1]


class CrossEntropyLabelSmooth(torch.nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            group_weight_decay.append(p)
        else:
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(
        group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(
        params=group_no_weight_decay, weight_decay=0.)]
    return groups


class StorageSaver:
    def __init__(self, architecture=None, saver_dictionary=None):
        if not architecture is None:
            self.architecture = architecture

        if not saver_dictionary is None:
            self.saver_dictionary = saver_dictionary

    def set_architecture(self, architecture):
        self.architecture = architecture

    def get_architecture(self):
        return self.architecture

    def set_saver_dataframe(self, saver_dataframe):
        self.saver_dictionary = saver_dataframe

    def get_saver_dictionary(self):
        return self.saver_dictionary


storage_saver = StorageSaver()


def get_device():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:' + ','.join([str(i) for i in GPU_ids]))
    return device


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoint(state, iters, tag=''):
    check_path(MODELS_PATH)
    filename = os.path.join(MODELS_PATH + "/{}checkpoint-{:06}.pth.tar".format(tag, iters))
    torch.save(state, filename)
    latest_filename = os.path.join(MODELS_PATH + "/{}checkpoint-latest.pth.tar".format(tag))
    torch.save(state, latest_filename)


def get_latest_model_path():
    check_path(MODELS_PATH)
    model_list = os.listdir(MODELS_PATH)
    if not model_list:
        return None, 0
    model_list.sort()
    latest_model = model_list[-2]
    iter = re.findall(r'\d+', latest_model)

    return MODELS_PATH + latest_model, int(iter[0])


def arr2str(arr):
    return ' '.join([str(i) for i in arr])


def write_results_csv(file_name, headers_name, row_data, operation='a'):
    if len(headers_name) != len(row_data):
        raise ValueError(CSV_DATA_MISMATCH)
    _write_data = list()

    if not os.path.exists(file_name):
        operation = 'w'
        _write_data.append(headers_name)

    _write_data.append(row_data)

    with open(file_name, operation) as f:
        writer = csv.writer(f)
        _ = [writer.writerow(i) for i in _write_data]
