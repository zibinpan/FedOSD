import tfedplat as fp
import numpy as np
import torch
import os


class FigRandBackdoor:
    def __init__(self, name="FigRandBackdoor", color=255/1.5, dataloader=None, save_folder="", save_name="backdoor"):
        self.name = name
        self.color = color
        if self.color < 0 or self.color > 255:
            raise RuntimeError("Color must be between 0 and 255.")
        if dataloader is None:
            raise RuntimeError('dataloader must be provided.')
        self.image_size = dataloader.raw_data_shape
        self.input_data_shape = dataloader.input_data_shape
        self.target_class_num = dataloader.target_class_num
        self.save_folder = save_folder
        self.save_name = save_name
        
        self.x_top = -4
        self.y_top = -4
        self.x_len = 3
        self.y_len = 3
        
        self.watermark = np.zeros(self.image_size)
        pattern = np.round(np.random.rand(self.x_len, self.y_len))  
        self.watermark[:, self.x_top: self.x_top + self.x_len, self.y_top: self.y_top + self.y_len] = pattern
        
        self.mean = 0.5
        self.std = 0.2
        self.color = (self.color / 255 - self.mean) / self.std
        
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        file_path = self.save_folder + self.save_name + '.npy'
        np.save(file_path, self)

    def add_backdoor(self, dataset, attack_portion=0.8):
        
        self.watermark = self.watermark.reshape(self.input_data_shape)
        self.watermark = torch.Tensor(self.watermark)
        
        for [batch_x, batch_y] in dataset:
            attack_num = int(len(batch_y) * attack_portion)
            batch_x[:attack_num] = (1 - self.watermark) * batch_x[:attack_num] + self.watermark * self.color
            batch_y[:attack_num] = (batch_y[:attack_num] + self.target_class_num // 2) % self.target_class_num  
