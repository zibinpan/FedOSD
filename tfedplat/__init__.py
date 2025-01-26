# -*- coding: utf-8 -*-
import os

# import modules
# import base class
from tfedplat.Algorithm import Algorithm
from tfedplat.Client import Client
from tfedplat.DataLoader import DataLoader
from tfedplat.Module import Module
from tfedplat.Metric import Metric
from tfedplat.seed import setup_seed
# import metrics
from tfedplat.metric.Correct import Correct
from tfedplat.metric.MAE import MAE
from tfedplat.metric.RMSE import RMSE
from tfedplat.metric.Precision import Precision
from tfedplat.metric.Recall import Recall
# import models
from tfedplat.model.LeNet5 import LeNet5
from tfedplat.model.CNN import CNN_CIFAR10
from tfedplat.model.MLP import MLP
from tfedplat.model.NFResNet import NFResNet18, NFResNet50

# import algorithm
import tfedplat.algorithm
from tfedplat.algorithm.FedAvg.FedAvg import FedAvg
# unlearning
from tfedplat.algorithm.unlearning.UnlearnAlgorithm import UnlearnAlgorithm
from tfedplat.algorithm.unlearning.FedOSD import FedOSD


# import backdoors
from tfedplat.dataloaders.backdoors.FigRandBackdoor import FigRandBackdoor


# import dataloader
from tfedplat.dataloaders.separate_data import separate_data, create_data_pool
from tfedplat.dataloaders.DataLoader_cifar10_pat import DataLoader_cifar10_pat
from tfedplat.dataloaders.DataLoader_mnist_pat import DataLoader_mnist_pat
from tfedplat.dataloaders.DataLoader_fashion_pat import DataLoader_fashion_pat
from tfedplat.dataloaders.DataLoader_cifar100_pat import DataLoader_cifar100_pat


# get the path of the data folder
data_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)

# get the path of the pool folder
pool_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/pool/'
if not os.path.exists(pool_folder_path):
    os.makedirs(pool_folder_path)

# get the path of the model
model_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/model/'

# import Task
from tfedplat.Task import BasicTask
from tfedplat.task.unlearning.UnlearningTask import UnlearningTask