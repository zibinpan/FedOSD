
import tfedplat as fp
import numpy as np
import time
import torch


class FedAvg(fp.Algorithm):
    def __init__(self,
                 name='FedAvg',
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

    def run(self):
        
        batch_num = np.mean(self.get_clinet_attr('training_batch_num'))
        while not self.terminated():
            com_time_start = time.time()
            
            m_locals, _, g_locals = self.train()
            com_time_end = time.time()
            cal_time_start = time.time()
            
            
            self.weight_aggregate(m_locals)
            
            self.current_training_num += self.epochs * batch_num
            
            cal_time_end = time.time()
            self.communication_time += com_time_end - com_time_start
            self.computation_time += cal_time_end - cal_time_start
            