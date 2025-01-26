
import tfedplat as fp
import numpy as np
import torch
import copy


class Client:
    
    def __init__(self,
                 id=None,
                 module=None,
                 device=None,
                 train_setting=None,
                 dishonest=None,
                 *args,
                 **kwargs):
        self.id = id
        if module is not None:
            module = module
        self.module = module
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.train_setting = copy.deepcopy(train_setting)  
        self.test = Test(self.train_setting, self.device)  
        self.dishonest = dishonest
        self.local_training_data = None
        self.local_training_number = 0
        self.local_test_data = None
        self.local_test_number = 0
        self.training_batch_num = 0
        self.test_batch_num = 0
        self.step_type = self.train_setting['step_type']
        self.criterion = self.train_setting['criterion'].to(self.device)
        
        self.upload_loss = None  
        self.upload_module = None  
        self.upload_grad = None  
        self.upload_training_acc = None  
        
        self.test_module = None

    @staticmethod
    def create_client_list(ClientClass, data_loader, module, device, train_setting, dishonest, client_list=None, client_num=None, *args, **kwargs):
        if client_list is not None:
            client_num = len(client_list)
        elif client_list is None and client_num is not None:
            if client_num > data_loader.pool_size:
                client_num = data_loader.pool_size
            client_list = [ClientClass(i, module, device, train_setting, dishonest, *args, **kwargs) for i in range(client_num)]  
            data_loader.allocate(client_list)  
        elif client_num is None and client_list is None:
            raise RuntimeError('Both of client_num and client_list cannot be None or not None.')
        return client_list, client_num

    def update_data(self,
                    local_training_data,
                    local_training_number,
                    local_test_data,
                    local_test_number,
                    ):
        self.local_training_data = local_training_data
        self.local_training_number = local_training_number
        self.local_test_data = local_test_data
        self.local_test_number = local_test_number
        self.training_batch_num = len(local_training_data)
        self.test_batch_num = len(local_test_data)

    def free_memory(self):
        self.upload_loss = None
        self.upload_module = None
        self.upload_grad = None
        self.upload_training_acc = None

    def get_message(self, msg):

        return_msg = {}
        
        if msg['command'] == 'cal_loss':
            self.upload_loss = self.cal_loss(self.module)
            return return_msg
        if msg['command'] == 'cal_gradient_loss':
            epochs = msg['epochs']
            lr = msg['lr']
            target_module = msg['target_module']
            if self.step_type == 'sgd':
                self.cal_gradient_loss_sgd(epochs, lr, target_module)
            else:
                self.cal_gradient_loss(epochs, lr, target_module)
            return return_msg
        if msg['command'] == 'train':
            
            epochs = msg['epochs']
            lr = msg['lr']
            target_module = msg['target_module']
            
            if self.step_type == 'sgd':
                self.train_SGD(epochs, lr, target_module)
            elif self.step_type == 'bgd':
                self.train(epochs, lr, target_module)
            elif self.step_type == 'fgd':
                self.train_fgd(epochs, lr, target_module)
            return return_msg
        if msg['command'] == 'free_memory':
            self.free_memory()
            return return_msg
        if msg['command'] == 'test':
            
            self.test_module = msg['test_module']  
            if self.test_module == 'upload_module':
                self.test_module = self.upload_module if self.upload_module is not None else self.module
            
            self.test.run(self)
            return return_msg
        if msg['command'] == 'require_loss':
            
            return_msg['l_local'] = self.upload_loss
            return return_msg
        if msg['command'] == 'require_gradient_loss':
            
            return_grad = self.upload_grad
            return_loss = self.upload_loss
            
            if self.dishonest is not None:
                if self.dishonest['scaled_update'] is not None:
                    return_grad *= self.dishonest['scaled_update']
                if self.dishonest['zero_update'] is not None:
                    return_grad *= 0.0
                if self.dishonest['random_update'] is not None:
                    n = len(return_grad)
                    grad = torch.randn(n).float().to(self.device)
                    
                    return_grad = grad  
            return_msg['g_local'] = return_grad
            return_msg['l_local'] = return_loss
            return return_msg
        if msg['command'] == 'require_client_module':
            
            return_module = self.upload_module
            return_loss = self.upload_loss
            if self.dishonest is not None:
                if self.dishonest['scaled_update'] is not None:
                    return_module = (return_module - self.module) * self.dishonest['scaled_update'] + self.module
                if self.dishonest['zero_update'] is not None:
                    return_module = copy.deepcopy(self.module)
                if self.dishonest['random_update'] is not None:
                    """
                    拜占庭攻击。
                    """
                    model_params_span = return_module.span_model_params_to_vec()
                    n = len(model_params_span)
                    updates = torch.randn(n).float().to(self.device)
                    old_model_params_span = self.module.span_model_params_to_vec()
                    return_module_params = old_model_params_span + updates
                    return_module.reshape_vec_to_model_params(return_module_params)

            return_msg['m_local'] = return_module
            return_msg['l_local'] = return_loss
            return return_msg
        if msg['command'] == 'require_test_result':
            return_msg['metric_history'] = copy.deepcopy(self.test.metric_history)
            return return_msg
        if msg['command'] == 'require_attribute_value':
            attr = msg['attr']
            return_msg['attr'] = getattr(self, attr)
            return return_msg

    def cal_loss_one_batch(self, module, batch_x, batch_y):

        out = module.model(batch_x)
        loss = self.criterion(out, batch_y)
        return loss

    def cal_loss(self, target_module):

        target_module.model.train()
        total_loss = 0.0  
        with torch.no_grad():
            for step, [batch_x, batch_y] in enumerate(self.local_training_data):
                batch_x = fp.Module.change_data_device(batch_x, self.device)
                batch_y = fp.Module.change_data_device(batch_y, self.device)
                loss = self.cal_loss_one_batch(target_module, batch_x, batch_y)  
                loss = float(loss)
                total_loss += loss * batch_y.shape[0]  
            loss = total_loss / self.local_training_number
        return loss

    def cal_gradient_loss(self, epochs, lr=0.1, target_module=None):

        if target_module is None:
            target_module = self.module
        
        loss = self.cal_loss(target_module)  
        target_module.model.train()
        grad_mat = []  
        total_loss = 0  
        weights = []
        for step, [batch_x, batch_y] in enumerate(self.local_training_data):
            batch_x = fp.Module.change_data_device(batch_x, self.device)
            batch_y = fp.Module.change_data_device(batch_y, self.device)
            weights.append(batch_y.shape[0])
            loss2 = self.cal_loss_one_batch(target_module, batch_x, batch_y)  
            total_loss += loss2 * batch_y.shape[0]  
            
            target_module.model.zero_grad()
            loss2.backward()
            
            target_module.clip_grad_norm_on_model(target_module.model, self.train_setting['g_clip'])
            grad_vec = target_module.span_model_grad_to_vec()
            grad_mat.append(grad_vec)
        weights = torch.Tensor(weights).float().to(self.device)
        weights = weights / torch.sum(weights)
        
        grad_mat = torch.stack([grad_mat[i] for i in range(len(grad_mat))])
        
        g = weights @ grad_mat

        self.upload_grad = g
        self.upload_loss = float(loss)

    def cal_gradient_loss_sgd(self, epochs, lr=0.1, target_module=None):

        if target_module is None:
            target_module = self.module
        
        self.upload_module = copy.deepcopy(target_module)
        optimizer = self.train_setting['optimizer'].__class__(filter(lambda p: p.requires_grad, self.upload_module.model.parameters()), lr=lr)
        optimizer.defaults = copy.deepcopy(self.train_setting['optimizer'].defaults)
        loss = self.cal_loss(self.upload_module)
        
        self.upload_module.model.train()
        self.upload_module.model.to(self.device)
        for e in range(epochs):
            sample_idx = int(np.random.choice(len(self.local_training_data), 1))  
            [batch_x, batch_y] = self.local_training_data[sample_idx]
            batch_x = fp.Module.change_data_device(batch_x, self.device)
            batch_y = fp.Module.change_data_device(batch_y, self.device)
            
            loss2 = self.cal_loss_one_batch(self.upload_module, batch_x, batch_y)
            
            self.upload_module.model.zero_grad()
            loss2.backward()
            
            self.upload_module.clip_grad_norm_on_model(self.upload_module.model, self.train_setting['g_clip'])
            
            optimizer.step()
        g = (target_module.span_model_params_to_vec() - self.upload_module.span_model_params_to_vec()) / lr  
        
        self.upload_grad = g
        self.upload_loss = float(loss)

    def do_after_train_step(self):
        return

    def train_SGD(self, epochs, lr, target_module=None):

        if target_module is None:
            target_module = self.module
        
        self.upload_module = copy.deepcopy(target_module)
        optimizer = self.train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.upload_module.model.parameters()), lr=lr)
        optimizer.defaults = copy.deepcopy(self.train_setting['optimizer'].defaults)
        
        loss = self.cal_loss(self.upload_module)
        
        self.upload_loss = float(loss)
        
        self.upload_module.model.train()
        for e in range(epochs):
            sample_idx = int(np.random.choice(len(self.local_training_data), 1))  
            [batch_x, batch_y] = self.local_training_data[sample_idx]
            batch_x = fp.Module.change_data_device(batch_x, self.device)
            batch_y = fp.Module.change_data_device(batch_y, self.device)
            
            loss = self.cal_loss_one_batch(self.upload_module, batch_x, batch_y)
            
            self.upload_module.model.zero_grad()
            loss.backward()
            
            self.upload_module.clip_grad_norm_on_model(self.upload_module.model, self.train_setting['g_clip'])
            
            optimizer.step()
            
            self.do_after_train_step()

    def train(self, epochs, lr, target_module=None):

        if target_module is None:
            target_module = self.module
        
        self.upload_module = copy.deepcopy(target_module)
        optimizer = self.train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.upload_module.model.parameters()), lr=lr)
        optimizer.defaults = copy.deepcopy(self.train_setting['optimizer'].defaults)
        
        loss = self.cal_loss(self.upload_module)  
        
        self.upload_loss = float(loss)
        
        self.upload_module.model.train()
        for e in range(epochs):
            for step, [batch_x, batch_y] in enumerate(self.local_training_data):
                batch_x = fp.Module.change_data_device(batch_x, self.device)
                batch_y = fp.Module.change_data_device(batch_y, self.device)
                
                loss = self.cal_loss_one_batch(self.upload_module, batch_x, batch_y)
                
                self.upload_module.model.zero_grad()
                loss.backward()
                
                self.upload_module.clip_grad_norm_on_model(self.upload_module.model, self.train_setting['g_clip'])
                
                optimizer.step()
                
                self.do_after_train_step()

    def train_fgd(self, epochs, lr, target_module=None):

        if target_module is None:
            target_module = self.module
        
        self.upload_module = copy.deepcopy(target_module)
        optimizer = self.train_setting['optimizer'].__class__(
            filter(lambda p: p.requires_grad, self.upload_module.model.parameters()), lr=lr)
        optimizer.defaults = copy.deepcopy(self.train_setting['optimizer'].defaults)
        
        if epochs <= 0:
            raise RuntimeError('error in Client: epochs must > 0')
        
        loss = self.cal_loss(self.upload_module)
        
        self.upload_loss = float(loss)
        
        self.upload_module.model.train()
        for e in range(epochs):
            for step, [batch_x, batch_y] in enumerate(self.local_training_data):
                batch_x = fp.Module.change_data_device(batch_x, self.device)
                batch_y = fp.Module.change_data_device(batch_y, self.device)
                
                loss = self.cal_loss_one_batch(self.upload_module, batch_x, batch_y) / self.training_batch_num
                
                loss.backward()
            
            self.upload_module.clip_grad_norm_on_model(self.upload_module.model, self.train_setting['g_clip'])
            
            optimizer.step()
            
            self.do_after_train_step()


class Test:

    def __init__(self, train_setting, device):
        self.train_setting = train_setting
        self.device = device
        
        self.metric_history = {'training_loss': [], 'test_loss': [], 'local_test_number': 0, 'test_accuracy': []}

    def run(self, client):
        client.test_module.model.eval()
        criterion = self.train_setting['criterion'].to(self.device)
        
        self.metric_history['training_loss'].append(round(float(client.upload_loss), 4) if client.upload_loss is not None else None)
        
        metric_dict = {'test_loss': 0, 'correct': 0}
        
        correct_metric = fp.Correct()
        
        with torch.no_grad():
            
            self.metric_history['local_test_number'] = client.local_test_number
            for [batch_x, batch_y] in client.local_test_data:
                batch_x = fp.Module.change_data_device(batch_x, self.device)
                batch_y = fp.Module.change_data_device(batch_y, self.device)
                
                out = client.test_module.model(batch_x)  
                loss = criterion(out, batch_y).item()
                metric_dict['test_loss'] += float(loss) * batch_y.shape[0]  
                metric_dict['correct'] += correct_metric.calc(out, batch_y)
            
            self.metric_history['test_loss'].append(round(metric_dict['test_loss'] / client.local_test_number, 4))
            self.metric_history['test_accuracy'].append(100 * metric_dict['correct'] / client.local_test_number)  
