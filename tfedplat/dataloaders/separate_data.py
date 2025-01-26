import tfedplat as fp
import numpy as np
import random
import torch


def separate_data(data, train_prob, num_clients, target_class_num, item_classes_num, batch_size, alpha, niid=False, balance=False, partition=None):
    least_samples = batch_size / (1-train_prob)  
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        item_classes_num = target_class_num

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(target_class_num):
            idx_for_each_class.append(idxs[dataset_label == i])  

        class_num_per_client = [item_classes_num for _ in range(num_clients)]
        for i in range(target_class_num):
            
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
                selected_clients = selected_clients[:int(num_clients/target_class_num*item_classes_num)]

            num_all_samples = len(idx_for_each_class[i])
            
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            
            
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(num_per/10, least_samples/target_class_num), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        
        min_size = 0
        K = target_class_num
        N = len(dataset_label)

        while min_size < target_class_num:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client]==i))))

    del data

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def seperate_batch_data(input_data, target_data, batch_size):
    batch_data_indices_list = fp.DataLoader.separate_list(list(range(len(target_data))), batch_size)
    local_data = []
    for batch_data_indices in batch_data_indices_list:
        
        batch_input_data = input_data[batch_data_indices]
        
        batch_target_data = target_data[batch_data_indices]
        local_data.append([batch_input_data, batch_target_data])
    return local_data

def seperate_batch_data_single(data, batch_size):
    
    batch_data_indices_list = fp.DataLoader.separate_list(list(range(len(data))), batch_size)
    local_data = []
    for batch_data_indices in batch_data_indices_list:
        batch_data = data[batch_data_indices]
        local_data.append(batch_data)
    return local_data


def create_data_pool(X, y, pool_size, shuffle, train_prob, batch_size, target_class_num):
    
    data_pool = [{} for _ in range(pool_size)]
    
    for pool_idx in range(pool_size):
        input_data = torch.Tensor(X[pool_idx]).float()
        target_data = torch.Tensor(y[pool_idx]).long()
        if shuffle:
            indices = list(range(len(target_data)))
            random.shuffle(indices)
            input_data = input_data[indices]
            target_data = target_data[indices]
        
        training_input_data = input_data[:int(train_prob*len(target_data))]
        training_target_data = target_data[:int(train_prob*len(target_data))]
        test_input_data = input_data[int(train_prob*len(target_data)):]
        test_target_data = target_data[int(train_prob*len(target_data)):]
        
        data_pool[pool_idx]['local_training_data'] = seperate_batch_data(training_input_data, training_target_data, batch_size)
        data_pool[pool_idx]['local_training_number'] = len(training_target_data)
        data_pool[pool_idx]['data_name'] = str(pool_idx)
        
        data_pool[pool_idx]['local_test_data'] = seperate_batch_data(test_input_data, test_target_data, batch_size)
        data_pool[pool_idx]['local_test_number'] = len(test_target_data)

    return data_pool
