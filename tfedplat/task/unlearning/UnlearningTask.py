import tfedplat as fp
import numpy as np
import torch
import os
import copy
from torchvision.transforms import transforms
transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class UnlearningTask(fp.BasicTask):

    def __init__(self, name='UnlearningTask'):
        super().__init__(name)
        
        self.algorithm.save_folder = self.name + '/' + self.params['module'] + '/' + self.data_loader.nickname + '/UN' + str(self.params['unlearn_cn']) + '/E' + str(self.params['E']) + '/C' + str(self.params['C']) + '/' + self.params['algorithm'] + '/'
        unlearn_pretrain_flag = self.params['unlearn_pretrain']
        
        
        pretrained_model_folder = self.name + '/' + self.params['module'] + '/' + self.data_loader.nickname + '/UN' + str(self.params['unlearn_cn']) + '/E' + str(self.params['E']) + '/'
        self.algorithm.pretrained_model_folder = pretrained_model_folder
        if not unlearn_pretrain_flag:  
            model_path = pretrained_model_folder + f'seed{self.params["seed"]}_unlearn_task_pretrained_model.pth'
            if not os.path.exists(pretrained_model_folder):
                os.makedirs(pretrained_model_folder)
            if not os.path.isfile(model_path):
                raise RuntimeError(f'Please put the pretrained model in the path {model_path}.')
            self.algorithm.module.model.load_state_dict(torch.load(model_path))
            self.algorithm.init_model_params = self.algorithm.module.span_model_params_to_vec()  
            self.algorithm.model_params = self.algorithm.module.span_model_params_to_vec()  
        else:  
            self.algorithm.save_model = True
            self.algorithm.model_save_name = pretrained_model_folder + f'seed{self.params["seed"]}_unlearn_task_pretrained_model.pth'  
            
            if isinstance(self.algorithm, fp.UnlearnAlgorithm):
                raise RuntimeError(f'When setting unlearn_pretrain_flag=True, you cannot run unlearning FL algorithm.')
            
            self.algorithm.terminate_extra_execute = self.terminate_extra_execute
        
        self.params['unlearn_client_id_list'] = np.random.choice(self.algorithm.client_num, self.params['unlearn_cn'], replace=False).tolist()
        print('Unlearn clients:', self.params['unlearn_client_id_list'])
        self.algorithm.out_log += f"Unlearn clients: {self.params['unlearn_client_id_list']}"
        
        for client in self.algorithm.client_list:
            
            if client.id in self.params['unlearn_client_id_list']:
                setattr(client, "unlearn_flag", True)
                pretrain_attack_portion = 0.8 if unlearn_pretrain_flag else 1.0
                self.modify_client(client, pretrain_attack_portion)  
            else:
                setattr(client, "unlearn_flag", False)

    @staticmethod
    def terminate_extra_execute(alg):
        alg.__class__.__bases__[0].terminate_extra_execute(alg)  

    def modify_client(self, client, attack_portion):
        
        setattr(client, "local_backdoor_test_data", copy.deepcopy(client.local_training_data))
        setattr(client, "local_backdoor_test_number", client.local_training_number)
        
        backdoor = fp.FigRandBackdoor(dataloader=self.algorithm.data_loader,save_folder=self.algorithm.pretrained_model_folder + 'backdoors/', save_name=f'client_{client.id}_backdoor')
        backdoor.add_backdoor(client.local_training_data, attack_portion=attack_portion)
        backdoor.add_backdoor(client.local_backdoor_test_data, attack_portion=attack_portion)
        setattr(client, "backdoor_setting", backdoor)  
        
        client.local_test_data = copy.deepcopy(client.local_training_data)
        client.local_test_number = client.local_training_number
        
        client.test = self.ClientTest(self.algorithm.train_setting, self.algorithm.device)

    @staticmethod
    class ClientTest:
        def __init__(self, train_setting, device):
            self.train_setting = train_setting
            self.device = device
            
            self.metric_history = {'training_loss': [], 'test_loss': [], 'local_test_number': 0, 'test_accuracy': [], 'backdoor_test_loss': [], 'backdoor_test_accuracy': []}

        def run(self, client):
            client.test_module.model.eval()
            criterion = self.train_setting['criterion'].to(self.device)
            
            self.metric_history['training_loss'].append(float(client.upload_loss) if client.upload_loss is not None else None)
            
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
                
                metric_dict = {'test_loss': 0, 'correct': 0}
                for [batch_x, batch_y] in client.local_backdoor_test_data:
                    batch_x = fp.Module.change_data_device(batch_x, self.device)
                    batch_y = fp.Module.change_data_device(batch_y, self.device)
                    
                    out = client.test_module.model(batch_x)  
                    loss = criterion(out, batch_y).item()
                    metric_dict['test_loss'] += float(loss) * batch_y.shape[0]  
                    metric_dict['correct'] += correct_metric.calc(out, batch_y)
                
                self.metric_history['backdoor_test_loss'].append(round(metric_dict['test_loss'] / client.local_backdoor_test_number, 4))
                self.metric_history['backdoor_test_accuracy'].append(100 * metric_dict['correct'] / client.local_backdoor_test_number)  

    @staticmethod
    def outFunc(alg):
        unlearned_client_loss_list = []
        retained_client_loss_list = []
        for i, metric_history in enumerate(alg.metric_log['client_metric_history']):
            training_loss = metric_history['training_loss'][-1]
            if training_loss is None:
                continue
            if alg.client_list[i].unlearn_flag:
                unlearned_client_loss_list.append(training_loss)
            else:
                retained_client_loss_list.append(training_loss)
        
        unlearned_client_local_acc_list = []
        retained_client_local_acc_list = []
        for i, metric_history in enumerate(alg.metric_log['client_metric_history']):
            test_acc = metric_history['test_accuracy'][-1]
            if alg.client_list[i].unlearn_flag:
                unlearned_client_local_acc_list.append(test_acc)
            else:
                retained_client_local_acc_list.append(test_acc)
        
        unlearned_client_local_backdoor_acc_list = []
        for i, metric_history in enumerate(alg.metric_log['client_metric_history']):
            if alg.client_list[i].unlearn_flag:
                test_acc = metric_history['backdoor_test_accuracy'][-1]
                unlearned_client_local_backdoor_acc_list.append(test_acc)
        unlearned_client_local_acc_list = np.array(unlearned_client_local_acc_list)
        retained_client_local_acc_list = np.array(retained_client_local_acc_list)
        unlearned_client_local_backdoor_acc_list = np.array(unlearned_client_local_backdoor_acc_list)
        
        def cal_fairness(values):
            p = np.ones(len(values))
            fairness = np.arccos(values @ p / (np.linalg.norm(values) * np.linalg.norm(p)))
            return fairness
        unlearned_client_fairness = cal_fairness(unlearned_client_local_acc_list)
        retained_client_fairness = cal_fairness(retained_client_local_acc_list)

        
        out_log = ""
        out_log += alg.save_name + ' ' + alg.data_loader.nickname + '\n'
        out_log += 'Lr: ' + str(alg.lr) + '\n'
        out_log += 'round {}'.format(alg.current_comm_round) + ' training_num {}'.format(alg.current_training_num) + '\n'
        out_log += f'Unlearned Client Mean Global Test loss: {format(np.mean(unlearned_client_loss_list), ".6f")}' + '\n' if len(unlearned_client_loss_list) > 0 else ''
        out_log += f'Unlearned Client Local Test Acc: {format(np.mean(unlearned_client_local_acc_list/100), ".3f")}({format(np.std(unlearned_client_local_acc_list/100), ".3f")}), angle: {format(unlearned_client_fairness, ".6f")}, min: {format(np.min(unlearned_client_local_acc_list), ".6f")}, max: {format(np.max(unlearned_client_local_acc_list), ".6f")}' + '\n' if len(unlearned_client_local_acc_list) > 0 else ''
        out_log += f'ASR: {format(np.mean(unlearned_client_local_backdoor_acc_list/100), ".3f")}({format(np.std(unlearned_client_local_backdoor_acc_list/100), ".3f")}), min: {format(np.min(unlearned_client_local_backdoor_acc_list), ".6f")}, max: {format(np.max(unlearned_client_local_backdoor_acc_list), ".6f")}' + '\n' if len(unlearned_client_local_backdoor_acc_list) > 0 else ''
        out_log += f'Retained Client Mean Global Test loss: {format(np.mean(retained_client_loss_list), ".6f")}' + '\n' if len(retained_client_loss_list) > 0 else ''
        out_log += f'Retained Client Local Test Acc: {format(np.mean(retained_client_local_acc_list/100), ".3f")}({format(np.std(retained_client_local_acc_list/100), ".3f")}), angle: {format(retained_client_fairness, ".6f")}, min: {format(np.min(retained_client_local_acc_list), ".6f")}, max: {format(np.max(retained_client_local_acc_list), ".6f")}' + '\n'
        out_log += f'communication_time: {alg.communication_time}, computation_time: {alg.computation_time} \n'
        out_log += '\n'
        alg.out_log = out_log + alg.out_log
        print(str(alg.name))
        print(out_log)

    def read_params(self, return_parser=False):
        parser = super().read_params(return_parser=True)

        
        parser.add_argument('--unlearn_cn', help='unlearn client num', type=int, default=0)
        parser.add_argument('--unlearn_pretrain', help='pretrain the model before unlearning', type=str, default=False)
        parser.add_argument('--UR', help='Unlearning round, must be smaller than R', type=int, default=100)
        parser.add_argument('--r_lr', help='Learning rate in the post-training', type=float, default=-1)  

        try:
            if return_parser:
                return parser
            else:
                params = vars(parser.parse_args())
                
                if params['UR'] > params['R']:
                    raise RuntimeError('The parameter of UR must not be bigger than R.')
                return params
        except IOError as msg:
            parser.error(str(msg))


if __name__ == '__main__':
    my_task = UnlearningTask()
    my_task.run()
