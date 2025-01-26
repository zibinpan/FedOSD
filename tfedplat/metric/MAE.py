
import tfedplat as fp
import torch


class MAE(fp.Metric):
    def __init__(self):
        super().__init__(name='MAE')

    @staticmethod
    def calc(network_output, target):
        mae = torch.sum(torch.abs(network_output - target)) / len(target)
        return mae.item()
