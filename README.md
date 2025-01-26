# Code of the AAAI-25 paper
![image](https://github.com/user-attachments/assets/97b41151-d3ba-4275-94dc-d6cdc87f993b)


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
![image](https://github.com/user-attachments/assets/b94026b9-9a9e-409b-bdae-393b09dad0fd)

![image](https://github.com/user-attachments/assets/3d576e45-45b6-4333-9ba3-32019f29b8c9)

![image](https://github.com/user-attachments/assets/9d4188ed-f3b3-476d-a840-bac05fd2f917)

![image](https://github.com/user-attachments/assets/2319c9a4-c8ad-436f-a2a7-a2da2efc3015)

## Key Challenges to FU
![image](https://github.com/user-attachments/assets/9f554e07-a711-4ae7-8eed-09f9314b4b76)

# The Proposed FedOSD
![image](https://github.com/user-attachments/assets/4e487b64-95a2-4501-aa82-c89f69d59535)

## Unlearning Cross-Entropy (UCE) Loss
![image](https://github.com/user-attachments/assets/0525c0c9-4648-4a55-bcc9-29a6aa2968f4)

## Orthogonal Steepest Descent Direction
![image](https://github.com/user-attachments/assets/70321347-e3d8-4a52-849d-27288340cda1)

![image](https://github.com/user-attachments/assets/b9c92c8c-5de9-4446-8f64-583895819582)

## Model Reverting Issue
![image](https://github.com/user-attachments/assets/a4d41a84-9202-49cb-a84d-a3f2a440e71a)

# Experiments
![image](https://github.com/user-attachments/assets/c2850137-0976-4ee1-900a-f026882db461)

![image](https://github.com/user-attachments/assets/ede468f8-5f1c-48ed-bbf8-72378b5cd917)

![image](https://github.com/user-attachments/assets/8770ebc9-8fd7-47f0-ba3f-6bd36b37aa03)

![image](https://github.com/user-attachments/assets/e2a325a0-98fb-413b-97ac-3ed5f2e25578)

![image](https://github.com/user-attachments/assets/25ab047a-0b2e-4f6d-a593-7a3d605e869b)

# Conmunication
Welcome to our session during the conference time. We will be there on March 1, 12:00 - 14:30.

# Cite
Pan Z, Wang Z, Li C, et al. Federated Unlearning with Gradient Descent and Conflict Mitigation[J]. arXiv preprint arXiv:2412.20200, 2024.

The final citation comes after proceeding with the camera-ready.
