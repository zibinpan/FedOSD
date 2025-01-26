
import torch
import copy
from torch.autograd import Variable


class Module:


    def __init__(self, device, *args, **kwargs):
        self.name = 'Module'
        self.device = device
        self.input_require_shape = None
        self.ignore_head = False  
        self.Loc_reshape_list = None
        self.Loc_list = None
        self.model = None

    def generate_model(self, input_data_shape, target_class_num=None, *args, **kwargs):

        raise NotImplementedError

    def fix_Loc_list(self):

        pass

    def create_Loc_reshape_list(self):
        
        currentIdx = 0
        self.Loc_reshape_list = []  
        self.Loc_list = []  
        for i, p in enumerate(self.model.parameters()):
            param_num = p.numel()  
            Loc = torch.arange(currentIdx, currentIdx + param_num, 1)
            self.Loc_list.append(Loc)
            self.Loc_reshape_list.append(Loc.reshape(p.data.shape))  
            currentIdx += param_num
        self.fix_Loc_list()

    @staticmethod
    def change_data_device(data, device):

        new_data = None
        if type(data) == torch.Tensor:
            new_data = data.to(device)
        elif type(data) == tuple:
            new_data = []
            for item in data:
                item = item.to(device)
                new_data.append(item)
            new_data = tuple(new_data)
        elif type(data) == list:
            new_data = []
            for item in data:
                item = item.to(device)
                new_data.append(item)
        return new_data

    def __add__(self, other):

        if isinstance(other, int) and other == 0:
            return self
        if not isinstance(other, Module):
            raise TypeError
        res_module = copy.deepcopy(self)
        res_module.model.zero_grad()
        res_model_params = res_module.model.state_dict()
        other_params = other.model.state_dict()
        for layer in res_model_params.keys():
            res_model_params[layer] += other_params[layer]
        return res_module

    def __sub__(self, other):

        if isinstance(other, int) and other == 0:
            return self
        if not isinstance(other, Module):
            raise TypeError
        res_module = copy.deepcopy(self)
        res_module.model.zero_grad()
        res_model_params = res_module.model.state_dict()
        other_params = other.model.state_dict()
        for layer in res_model_params.keys():
            res_model_params[layer] -= other_params[layer]
        return res_module

    def __mul__(self, other):

        res_module = copy.deepcopy(self)
        res_module.model.zero_grad()
        res_model_params = res_module.model.state_dict()
        if not isinstance(other, Module):
            for k in res_model_params.keys():
                res_model_params[k] *= other
        else:
            other_params = other.model.state_dict()
            for k in res_module.model.state_dict().keys():
                res_model_params[k] *= other_params[k]
        return res_module

    def __rmul__(self, other):

        return self * other

    def __pow__(self, power):
        return self._model_norm(power)

    def dot(self, other):

        res = 0.0
        md1 = self.model.state_dict()
        md2 = other.model.state_dict()
        for layer in md1.keys():
            if md1[layer] is None:
                continue
            
            
            res += torch.sum(md1[layer] * md2[layer])
        return res

    def dot_vec(self, a, b):
        sums = 0.0
        for layer in self.Loc_list:
            a_layer = a[layer]
            b_layer = b[layer]
            sums += a_layer @ b_layer
        return sums

    def L2_norm_square(self):

        model_params = self.model.state_dict()
        res = 0.0
        for k in model_params.keys():
            if model_params[k] is None: continue
            if model_params[k].dtype not in [torch.float, torch.float32, torch.float64]:
                continue
            res += torch.sum(torch.pow(model_params[k], 2))
        return res

    def norm(self, p=2):
        return self**p

    def _model_norm(self, p):

        res = 0.0
        md = self.model.state_dict()
        for layer in md.keys():
            if md[layer] is None:
                continue
            if md[layer].dtype not in [torch.float, torch.float32, torch.float64]:
                continue
            res += torch.sum(torch.pow(md[layer], p))
        return torch.pow(res, 1.0 / p)

    @staticmethod
    def model_sum(module_list):

        res_module = copy.deepcopy(module_list[0])
        res_model_params = res_module.model.state_dict()
        for module in module_list[1:]:
            model_params = module.model.state_dict()
            for k in model_params.keys():
                res_model_params[k] += model_params[k]
        return res_module

    def clone_from(self, module):

        self.model.load_state_dict(module.model.state_dict())

    def span_model_grad_to_vec(self):

        grad_vec = []
        for p in self.model.parameters():
            if p.grad is not None:
                flat = p.grad.data.clone().flatten()
                grad_vec.append(Variable(flat, requires_grad=False))
        grad_vec = torch.cat(grad_vec)
        return grad_vec

    def reshape_vec_to_model_grad(self, vec):
        for i, p in enumerate(self.model.parameters()):
            if p.grad is not None:
                p.grad = vec[self.Loc_reshape_list[i]]

    def span_model_params_to_vec(self):

        param_vec = []
        model_params = self.model.state_dict()
        for layer in model_params.keys():
            flat = model_params[layer].clone().flatten()
            param_vec.append(Variable(flat, requires_grad=False))
        param_vec = torch.cat(param_vec)
        return param_vec

    def reshape_vec_to_model_params(self, vec):

        for i, p in enumerate(self.model.parameters()):
            p.data = vec[self.Loc_reshape_list[i]]

    def print_layer_name_numel(self):
        params = self.model.state_dict()
        for layer in params.keys():
            print(layer)
            print(params[layer].numel())

    @staticmethod
    def clip_grad_norm_on_model(model, max_norm, norm_type="2"):
        
        if max_norm > 0:
            if norm_type == "inf":
                norm_type = torch.inf
            elif norm_type == "-inf":
                norm_type = -torch.inf
            else:
                norm_type = float(norm_type)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type)

    def clip_grad_norm_on_vector(self, g, max_norm, norm_type="2"):

        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if max_norm > 0:
            if norm_type == "inf":
                norm_type = torch.inf
            elif norm_type == "-inf":
                norm_type = -torch.inf
            else:
                norm_type = float(norm_type)
            grads = []
            for i in range(len(self.Loc_reshape_list)):
                grads.append(g[self.Loc_reshape_list[i]])
            torch.nn.utils.clip_grad_norm_(grads, max_norm, norm_type)
            for i in range(len(self.Loc_reshape_list)):
                g[self.Loc_reshape_list[i]] = grads[i]

if __name__ == '__main__':
    
    x = torch.tensor([102.0, 155.0], requires_grad=True)
    y = torch.tensor([201.0, 221.0], requires_grad=True)

    z = x ** 3 + y ** 4
    z = z.sum()
    
    z.backward()

    print(f"gradient of x is:{x.grad}")
    print(f"gradient of y is:{y.grad}")
    
    torch.nn.utils.clip_grad_norm_([x, y], max_norm=200, norm_type=-torch.inf)
    
    print("---clip_grad---")
    print(f"clip_grad of x is:{x.grad}")
    print(f"clip_grad of y is:{y.grad}")
