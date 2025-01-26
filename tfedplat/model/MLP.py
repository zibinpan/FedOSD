
import tfedplat as fp
import torch
import torch.nn.functional as F


class MLP(fp.Module):

    def __init__(self, device, *args, **kwargs):
        super(MLP, self).__init__(device)
        self.name = 'MLP'
        self.model = None
        
        self.input_require_shape = [-1]
        
        self.ignore_head = False

    def generate_model(self, input_data_shape, target_class_num=None, *args, **kwargs):
        self.model = MLP_Model(self.ignore_head, input_data_shape, target_class_num).to(self.device)
        self.create_Loc_reshape_list()

    def fix_Loc_list(self):
        
        temp = self.Loc_list
        temp[0] = torch.hstack([temp[0], temp[1]])
        temp[1] = None
        temp[2] = torch.hstack([temp[2], temp[3]])
        temp[3] = None
        temp[4] = torch.hstack([temp[4], temp[5]])
        temp[5] = None
        self.Loc_list = []
        for item in temp:
            if item is not None:
                self.Loc_list.append(item)


class MLP_Model(torch.nn.Module):
    def __init__(self, ignore_head, input_data_shape, target_class_num):
        super(MLP_Model, self).__init__()
        self.ignore_head = ignore_head
        input_dim = input_data_shape[0]
        self.fc1 = torch.nn.Linear(input_dim, 400)
        self.fc2 = torch.nn.Linear(400, 400)
        self.predictor = torch.nn.Linear(400, target_class_num)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        if not self.ignore_head:
            x = self.predictor(x)
        return x
