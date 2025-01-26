
import tfedplat as fp
import torch


class Correct(fp.Metric):
    def __init__(self):
        super().__init__(name='correct')

    @staticmethod
    def calc(network_output, target):
        _, predicted = torch.max(network_output, -1)
        return predicted.eq(target).sum().item()
