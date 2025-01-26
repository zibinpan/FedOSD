
import tfedplat as fp
import time
import torch
import torch.nn.functional as F


class FedOSD(fp.UnlearnAlgorithm):
    def __init__(self,
                 name='FedOSD',
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
        self.initial_g_norm = None

    def cal_psedoinverse(self, matrix):
        U, s, V = torch.svd(matrix)  
        primary_sigma_indices = torch.where(s >= 1e-6)[0]
        s[primary_sigma_indices] = 1 / s[primary_sigma_indices]
        S = torch.diag(s)
        psedoinverse = V @ S @ U.T
        return psedoinverse

    def get_nearest_oth_d(self, gr_locals, gu):
        start_time = time.time()

        A = gr_locals
        
        A_T = A.T
        c = gu
        
        AAT_1 = self.cal_psedoinverse(A @ A_T)  
        
        Ac= A @ c.reshape(-1, 1)
        
        AAT_1_Ac = AAT_1 @ Ac
        
        d = c - (A_T @ AAT_1_Ac).reshape(-1)

        cal_time = time.time() - start_time
        return d, cal_time

    def train_a_round(self):
        com_time_start = time.time()
        
        m_locals, l_locals, g_locals = self.train()
        
        com_time_end = time.time()

        if not self.recovery_stage:
            gu_locals = []
            gr_locals = []
            for idx, client in enumerate(self.online_client_list):
                if client.unlearn_flag:
                    gu_locals.append(g_locals[idx])
                else:
                    gr_locals.append(g_locals[idx])
            gu_locals = torch.vstack(gu_locals)
            gr_locals = torch.vstack(gr_locals)
            
            weights = torch.Tensor([1 / len(gu_locals)] * len(gu_locals)).float().to(self.device)
            gu = weights @ gu_locals
            gu_norm = torch.norm(gu)
            g_norm = gu_norm
            
            d, cal_time = self.get_nearest_oth_d(gr_locals, gu)
            
            d = d / torch.norm(d) * g_norm
            
            if self.test_conflicts:
                self.stat_update_conflict(d, gr_locals)
            
            lr = self.lr
            self.update_module(self.module, self.optimizer, lr, d)
        else:
            
            model_params = self.module.span_model_params_to_vec()  
            ga = model_params - self.init_model_params  
            ga_norm = torch.norm(ga)
            
            for idx in range(len(g_locals)):
                grad = g_locals[idx]
                g_norm = torch.norm(grad)
                g_locals[idx] = grad - grad @ ga / ga_norm**2 * ga
                g_locals[idx] = g_locals[idx] / torch.norm(g_locals[idx]) * g_norm

            cal_time_start = time.time()
            
            weights = torch.Tensor([1 / len(l_locals)] * len(l_locals)).float().to(self.device)
            d = weights @ g_locals
            
            lr = self.lr
            self.update_module(self.module, self.optimizer, lr, d)

            cal_time = time.time() - cal_time_start
        
        self.communication_time += com_time_end - com_time_start
        self.computation_time += cal_time

    def run(self):
        
        for client in self.client_list:
            if client.unlearn_flag:
                client.criterion = self.UnLearningCELoss()
        
        while not self.terminated():
            self.train_a_round()
