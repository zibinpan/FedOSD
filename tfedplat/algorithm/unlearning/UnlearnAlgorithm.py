
import tfedplat as fp
import numpy as np
import os
import torch
import torch.nn.functional as F


class UnlearnAlgorithm(fp.Algorithm):
    def __init__(self,
                 name='UnlearnAlgorithm',
                 data_loader=None,
                 module=None,
                 device=None,
                 train_setting=None,
                 client_num=None,
                 client_list=None,
                 online_client_num=None,
                 save_model=False,
                 max_comm_round=0,
                 max_training_num=0,
                 epochs=1,
                 save_name=None,
                 outFunc=None,
                 write_log=True,
                 dishonest=None,
                 test_conflicts=False,
                 params=None,
                 *args,
                 **kwargs):
        
        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num, save_model, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, test_conflicts, params)
        
        self.max_unlearn_round = params['UR'] if params is not None else int(0.5 * self.max_comm_round)  
        self.r_lr = params['r_lr']
        self.recovery_stage = False  
        self.init_model_params = None  
        self.algorithm_log['distance_to_init_model'] = []  
        
        self.total_conflict_angle = 0
        self.total_conflict_num = 0

    def reinit(self):
        super().reinit()
        self.recovery_stage = False

    class UnLearningCELoss:
        def __init__(self, ignore_index=-100, reduction='mean'):
            self.ignore_index = ignore_index
            self.reduction = reduction

        def __call__(self, pred, target):
            class_num = int(pred.shape[1])
            
            ignore_indices = torch.where(target == self.ignore_index)[0]
            if len(ignore_indices) > 0:
                target[ignore_indices] = 0  
                target_enc = F.one_hot(target, class_num)
                target_enc[ignore_indices, 0] = 0  
            else:
                target_enc = F.one_hot(target, class_num)
            
            pred = F.softmax(pred, dim=-1)
            if self.reduction == 'none':
                loss = -torch.sum(torch.log(1.0 - pred / 2) * target_enc, dim=1)
            elif self.reduction == 'mean':
                loss = -torch.mean(torch.sum(torch.log(1.0 - pred / 2) * target_enc, dim=1))
            elif self.reduction == 'sum':
                loss = -torch.sum(torch.sum(torch.log(1.0 - pred / 2) * target_enc, dim=1))
            else:
                loss = None
            return loss

    def stop_unlearn_and_start_recovery(self):
        def save_model_log():
            self.recovery_stage = True
            
            if self.r_lr > 0:
                self.initial_lr = self.r_lr
            
            if self.save_model:
                self.module.model.to("cpu")
                if not os.path.exists(self.save_folder):
                    os.makedirs(self.save_folder)
                model_save_name = self.save_folder + self.save_name + '_unlearn.pth'
                torch.save(self.module.model.state_dict(), model_save_name)
                print('Saved the unlearned model.')
                self.module.model.to(self.device)  
            
            if self.write_log:
                self.save_log(save_name=self.save_name + '_unlearn')
                print('Saved log in the unlearning stage.')

        if self.current_comm_round > self.max_unlearn_round:  
            save_model_log()
        else:
            unlearned_client_local_backdoor_acc_list = []
            for i, metric_history in enumerate(self.metric_log['client_metric_history']):
                if self.client_list[i].unlearn_flag:
                    test_acc = metric_history['backdoor_test_accuracy'][-1]
                    unlearned_client_local_backdoor_acc_list.append(test_acc)
            unlearned_client_local_backdoor_acc_list = np.array(unlearned_client_local_backdoor_acc_list)

    @staticmethod
    def terminate_extra_execute(self):
        super().terminate_extra_execute(self)  
        
        ga_norm = torch.norm(self.model_params - self.init_model_params)  
        print(f'distance: {ga_norm}')
        self.out_log = f'distance: {round(float(ga_norm), 4)}' + '\n' + self.out_log
        self.algorithm_log['distance_to_init_model'].append(round(float(ga_norm), 4))

    def terminated(self):
        
        terminated_flag = super().terminated()
        
        if not self.recovery_stage:
            print('Unlearning stage')
            self.stop_unlearn_and_start_recovery()
        
        if self.recovery_stage:
            print('Post-training stage')
            online_client_list = []
            for client in self.online_client_list:
                if not client.unlearn_flag:  
                    online_client_list.append(client)
            self.online_client_list = online_client_list
        return terminated_flag

    def stat_update_conflict(self, d, gr_locals):
        out_log = ""
        
        conflict_angle = 0
        conflict_client_num = 0
        for gr_local in gr_locals:
            angle = self.cal_vec_angle(gr_local, d)
            if angle > 90:
                conflict_angle += (angle - 90) / 90
                conflict_client_num += 1
        self.total_conflict_angle += conflict_angle
        self.total_conflict_num += conflict_client_num
        if conflict_client_num > 0:
            out_log += f'mean conflict coefficient: {conflict_angle / conflict_client_num} \n'
            out_log += f'conflict num: {conflict_client_num} \n'
        if self.total_conflict_num > 0:
            out_log += f'historical mean conflict coefficient: {self.total_conflict_angle / self.total_conflict_num} \n'
            out_log += f'historical mean conflict num: {self.total_conflict_num / self.current_comm_round} \n'
            self.out_log = out_log + self.out_log
        print(out_log)
