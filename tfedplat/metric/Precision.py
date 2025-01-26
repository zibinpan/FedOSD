
import tfedplat as fp


class Precision(fp.Metric):
    def __init__(self):
        super().__init__(name='precision')

    @staticmethod
    def calc(pred, target):
        
        true_positive = ((target * pred) > .1).int().sum(axis=-1)
        return (true_positive / (pred.sum(axis=-1) + 1e-13)).sum().item()  
