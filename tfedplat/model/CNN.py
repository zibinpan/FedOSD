
import tfedplat as fp
import torch
import torch.nn as nn


class CNN_CIFAR10(fp.Module):

    def __init__(self, device, *args, **kwargs):
        super(CNN_CIFAR10, self).__init__(device)
        self.name = 'CNN_CIFAR10'
        self.model = None
        
        self.input_require_shape = [3, -1, -1]
        
        self.ignore_head = False

    def generate_model(self, input_data_shape, target_class_num=None, *args, **kwargs):
        self.model = CNN(self.ignore_head, input_data_shape, target_class_num).to(self.device)
        self.create_Loc_reshape_list()

    def fix_Loc_list(self):
        
        temp = self.Loc_list
        temp[0] = torch.hstack([temp[0], temp[1]])
        temp[1] = None
        temp[2] = torch.hstack([temp[2], temp[3]])
        temp[3] = None
        temp[4] = torch.hstack([temp[4], temp[5]])
        temp[5] = None
        temp[6] = torch.hstack([temp[6], temp[7]])
        temp[7] = None
        temp[8] = torch.hstack([temp[8], temp[9]])
        temp[9] = None
        self.Loc_list = []
        for item in temp:
            if item is not None:
                self.Loc_list.append(item)


class CNN(nn.Module):
    def __init__(self, ignore_head, input_data_shape, target_class_num):
        super(CNN, self).__init__()
        self.ignore_head = ignore_head
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1600, 384),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.predictor = nn.Linear(192, target_class_num)

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        x = self.decoder(x)
        if not self.ignore_head:
            x = self.predictor(x)
        return x
