# Code of the AAAI-25 paper

![image](https://github.com/user-attachments/assets/70824376-ea6a-4fe7-ad91-2031fbf34872)

# How to run the code
First, run the following command to train a FL global model as the original model for unlearning experiments.
```
/usr/bin/python run_UnlearningTask.py --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedAvg --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --R 2000 --E 1 --lr 0.05 --decay 0.999 --step_type bgd --unlearn_cn 1 --unlearn_pretrain True --save_model True
```

Next, execute the following command to run the unlearning algorithm.
```
/usr/bin/python run_UnlearningTask.py --seed 1 --device 0 --module CNN_CIFAR10 --algorithm FedOSD --dataloader DataLoader_cifar10_pat --N 10 --NC 2 --balance True --B 200 --C 1.0 --R 200 --UR 100 --E 1 --decay 0.999 --step_type bgd --unlearn_cn 1 --save_model True --lr 0.0004 --r_lr 1e-6
```

# Background
![image](https://github.com/user-attachments/assets/ac79d22a-c63c-488a-814b-599808564d32)

![image](https://github.com/user-attachments/assets/1a2ee6c2-d6f7-444e-8013-bcf3a182ea4a)

![image](https://github.com/user-attachments/assets/177ab85b-740d-499e-a2c8-3a744f832bf6)

![image](https://github.com/user-attachments/assets/eb621a40-61f2-4bf9-920d-efcbc85b4ab8)

## Key Challenges to FU
![image](https://github.com/user-attachments/assets/529b95eb-eb77-4725-8c7f-36bec0a3ba27)

# The Proposed FedOSD
![image](https://github.com/user-attachments/assets/357adb7d-f4c1-4ade-843a-4b6493b4a1a6)

## Unlearning Cross-Entropy (UCE) Loss
![image](https://github.com/user-attachments/assets/df9b0a97-6259-4e74-93ed-ffa55bdb0c5f)

## Orthogonal Steepest Descent Direction
![image](https://github.com/user-attachments/assets/d25175ce-55a7-48fd-9629-e235fe8810cb)

![image](https://github.com/user-attachments/assets/4f2b0f34-8d8f-432c-ae57-d3e3b3df98dd)

## Model Reverting Issue
![image](https://github.com/user-attachments/assets/ceeb19b3-e2d5-410d-aca3-b1d1fbc3bde5)

# Experiments
![image](https://github.com/user-attachments/assets/d99268cf-f2ff-4bf0-8c1b-4b6193920209)

![image](https://github.com/user-attachments/assets/7b99db90-451e-4799-8aeb-fead9dc16441)

![image](https://github.com/user-attachments/assets/c59ef391-09a6-4597-ba1a-07eed52678fc)

![image](https://github.com/user-attachments/assets/aefaa15c-7854-44c1-829a-b6853827bdb5)

# Conmunication
Welcome to our session during the conference time. We will be there on March 1, 12:00 - 14:30.

# Cite
Pan Z, Wang Z, Li C, et al. Federated Unlearning with Gradient Descent and Conflict Mitigation[J]. arXiv preprint arXiv:2412.20200, 2024.

The final citation comes after proceeding with the camera-ready.
