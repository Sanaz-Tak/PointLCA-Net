#!/usr/bin/env python3
"""
N-MNIST PointLCA Decoder
Extracts sparse representations from trained PointNet models using Locally Competitive Algorithm.

Usage Example:
    python Run-PT-NMNIST-Decoder.py \
        --model_path ./log/classification/2025-08-13_08-10/checkpoints/best_model.pth \
        --data_dir ./processed_data/nmnist \
        --num_point 1024

Required Arguments:
    --model_path: Path to your trained PointNet checkpoint
    --data_dir: Directory containing processed N-MNIST H5 files
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import time
import os
import sys
import argparse
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

def parse_args():
    parser = argparse.ArgumentParser('N-MNIST PointLCA Decoder')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained PointNet model checkpoint')
    parser.add_argument('--data_dir', default='./processed_data/nmnist', 
                       help='Directory with processed H5 files')
    parser.add_argument('--dictionary_size', type=int, default=60000,
                       help='Size of LCA dictionary (N-MNIST optimal: 60000)')
    parser.add_argument('--neuron_iterations', type=int, default=100, 
                       help='Number of LCA neuron iterations')
    parser.add_argument('--lambda_sparsity', type=float, default=0.2, 
                       help='Sparsity coefficient (lambda)')
    parser.add_argument('--lr_dictionary', type=float, default=0.001, 
                       help='Learning rate for dictionary update')
    parser.add_argument('--lr_neuron', type=float, default=0.001, 
                       help='Learning rate for neuron update')
    parser.add_argument('--batch_size', type=int, default=24, 
                       help='Batch size for processing')
    return parser.parse_args()


class LCA:
    """Locally Competitive Algorithm for sparse coding """
    
    def __init__(self, feature_size, dictionary_num, UPDATE_DICT, dictionary_iter, 
                 neuron_iter, lr_dictionary, lr_neuron, landa):
        self.feature_size = feature_size
        self.dict_num = dictionary_num
        self.UPDATE_DICT = UPDATE_DICT
        self.dict_iter = dictionary_iter
        self.neuron_iter = neuron_iter
        self.lr_dict = lr_dictionary
        self.lr_neuron = lr_neuron
        self.landa = landa

        self.dictionary = None
        self.data = None
        self.input = None
        self.a = None
        self.u = None

    def lca_update(self, n, phi, G, device):
        u_list = [torch.zeros([1, self.dict_num]).to(device)]
        a_list = [self.threshold(u_list[0], 'soft', True, self.landa).to(device)]

        if n == 0:
            dict = self.dictionary.reshape(self.dict_num, -1)
            phi = dict.T
            phi = phi.to(device)
            I = torch.eye(self.dict_num).to(device)
            G = torch.mm(phi.T, phi) - I
            G = G.to(device)
            input = self.input.detach().reshape(-1)
        elif n == 1:
            input = self.input.reshape(fn_train, 3 * self.feature_size * self.feature_size)
        elif n == 2:
            input = self.input.reshape(fn_test, 3 * self.feature_size * self.feature_size)
        else:
            input = self.input.reshape(100, -1)

        S = input.T

        b = torch.matmul(S.T, phi)
        for t in range(self.neuron_iter):
            u = self.neuron_update(u_list[t], a_list[t], b, G)
            u_list.append(u)
            a = self.threshold(u, 'soft', True, self.landa)
            a_list.append(a)

        self.a = a_list[-1]
        self.u = u_list[-1]

    def loss(self):
        # only consider the reconstruction loss
        s = self.input.reshape(512 * 7 * 7, 1)
        phi = self.dictionary.reshape(self.dict_num, -1).T
        a = self.a
        residual = s - torch.mm(phi, a.T)
        # l2 loss
        approximation_loss = .5 * torch.linalg.norm(residual, 'fro')
        sparsity_loss = self.landa * torch.sum(torch.abs(a))
        loss = approximation_loss + sparsity_loss
        print('Loss: {:.2f}'.format(loss.item()), 'approximation loss: {:.2f}'.format(approximation_loss.item()), 'sparsity loss: {:.2f}'.format(sparsity_loss.item()))
        return loss

    def dict_update(self):
        phi = self.dictionary.reshape(self.dict_num, -1).T
        phi = phi.to(device)        
        S = self.input.reshape(-1, 1)
        d_phi = torch.matmul((S.reshape(-1, 50)-torch.matmul(phi, self.a.T)), self.a)
        d_dict = d_phi.T.reshape([self.dict_num, 1000])
        d_dict = d_dict.cpu()
        self.dictionary = self.dictionary + d_dict * self.lr_dict
        return

    def threshold(self, u, type, rectify, landa):
        u_zeros = torch.zeros_like(u)
        if type == 'soft':
            if rectify:
                # define spikes
                a_out = torch.where(torch.greater(u, landa), u - landa,
                                 u_zeros)
            else:
                a_out = torch.where(torch.ge(u, landa), u - landa,
                                    torch.where(torch.le(u, - landa), u + landa,
                                                u_zeros))
        elif type == 'hard':
            if rectify:
                a_out = torch.where(
                    torch.gt(u, landa),
                    u,
                    u_zeros)
            else:
                a_out = torch.where(
                    torch.ge(u, landa),
                    u,
                    torch.where(
                        torch.le(u, -landa),
                        u,
                        u_zeros))
        else:
            assert False, (f'Parameter thresh_type must be "soft" or "hard", not {type}')
        return a_out

    def neuron_update(self, u_in, a_in, b, G):
        du = b - torch.mm(a_in, G) - u_in
        u_out = u_in + self.lr_neuron * du
        return u_out

    def reconstruct(self):
        recon = torch.matmul(self.a, self.dictionary.reshape([self.dict_num,-1]))
        recon = torch.sum(recon, axis=0)
        return recon


def normalize(M):
    sigma = torch.sum(M * M)
    return M / torch.sqrt(sigma)


class ProcessedNMNISTDataset:
    """Dataset class for processed N-MNIST H5 files"""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        
        # Load processed H5 file
        h5_file = os.path.join(data_dir, f'nmnist_{split}.h5')
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"Processed N-MNIST {split} file not found: {h5_file}")
        
        with h5py.File(h5_file, 'r') as f:
            self.data = f['data'][:]      # Shape: (N, 3, 1024)
            self.labels = f['label'][:]    # Shape: (N,)
        
        print(f"Loaded {split} dataset: {len(self.data)} samples, shape: {self.data.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), int(self.labels[idx])


def main():
    args = parse_args()
    
    # Set random seed for reproducibility (like reference project)
    torch.manual_seed(1234)
    
    # Set device
    if not args.use_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Global variables for LCA
    global fn_train, fn_test
    fn_train = 60000
    fn_test = 10000
    class_num = 10
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = ProcessedNMNISTDataset(args.data_dir, 'train')
    test_dataset = ProcessedNMNISTDataset(args.data_dir, 'test')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=fn_train, shuffle=True, num_workers=10, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=fn_test, shuffle=False, num_workers=10)
    
    # Load trained PointNet model
    print(f"Loading trained model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    # Import the model module
    model_dir = os.path.dirname(args.model_path)
    log_dir = os.path.dirname(model_dir)
    logs_dir = os.path.join(log_dir, 'logs')
    
    if os.path.exists(logs_dir):
        model_files = [f for f in os.listdir(logs_dir) if f.endswith('.txt')]
        if model_files:
            model_name = model_files[0].split('.')[0]
            print(f"Loading model: {model_name}")
            
            # Import the model
            sys.path.append(logs_dir)
            model_module = __import__(model_name)
            
            # Create model instance
            classifier = model_module.get_model(10, normal_channel=False)  # N-MNIST has 10 classes
            
            # Load trained weights
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
            classifier.load_state_dict(checkpoint['model_state_dict'])
            classifier = classifier.to(device)
            
            print("Model loaded successfully!")
        else:
            raise FileNotFoundError("No model files found in logs directory")
    else:
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")
    
    # Use the feature extractor like in reference
    res_model = classifier.feat
    res_model.eval()
    
    # Get training data
    dataiter = iter(train_loader)
    data_train, labels_train = next(dataiter)
    print('Training data done')
    
    # Get test data
    dataiter = iter(test_loader)
    data_test, labels_test = next(dataiter)
    print('Test data done')
    
    print("dataset_train shape:", data_train.shape, data_train.device)
    print("labels_train shape:", len(labels_train))
    
    # Initialize LCA
    lca = LCA(
        feature_size=1024,
        dictionary_num=args.dictionary_size,
        UPDATE_DICT=False,
        dictionary_iter=1,
        neuron_iter=args.neuron_iterations,
        lr_dictionary=args.lr_dictionary,
        lr_neuron=args.lr_neuron,
        landa=args.lambda_sparsity
    )
    
    # Extract all feature maps for dictionary initialization
    print("Extracting feature maps for dictionary initialization...")
    all_feature_maps = torch.tensor([]).to(device)
    
    # Handle case where dict_num is smaller than 1000
    if lca.dict_num <= 1000:
        batch_size = lca.dict_num
        with torch.no_grad():
            feature_maps = res_model(data_train[:batch_size].to(device))[0].to(device)
            all_feature_maps = feature_maps.reshape(-1, 1024)
    else:
        for i in range(lca.dict_num//1000):
            with torch.no_grad():
                feature_maps = res_model(data_train[i*1000:(i+1)*1000].to(device))[0].to(device)
                all_feature_maps = torch.cat((all_feature_maps, feature_maps.reshape(-1, 1024)), dim=0)
    
    print('all_feature_maps:', all_feature_maps.shape)
    
    # Initialize dictionary
    lca.dictionary = torch.zeros(lca.dict_num, 1024)
    for i in range(lca.dict_num):
        lca.dictionary[i] = normalize(all_feature_maps[i].detach())
    
    del all_feature_maps
    
    # Compute G matrix
    dict = lca.dictionary.reshape(lca.dict_num, -1)
    phi = dict.T
    I = torch.eye(lca.dict_num)
    G = torch.mm(phi.T, phi) - I
    phi = phi.to(device)
    G = G.to(device)
    
    # Training a neural network on top of LCA
    print("Training neural network on LCA features...")
    for i in range(600):
        lca.input = res_model(data_train[i*100:(i+1)*100].to(device))[0].to(device)
        lca.lca_update(20, phi, G, device)
        a = lca.a.clone().detach().type(torch.float).to('cpu')
        if i == 0:
            a_all = a
        else:
            a_all = torch.cat((a_all, a), 0)
    
    print("a_all(train): ", a_all.shape)
    
    # Evaluate training accuracy
    indices = torch.argmax(a_all, dim=1).to('cpu')
    print(f'Training accuracy (max) = {sum(labels_train[indices] - labels_train == 0) / fn_train}')
    
    indices_dict = {}
    for digit in range(class_num):
        indices_dict[digit] = torch.nonzero(labels_train[0:lca.dict_num] == digit).squeeze()
    
    max_indices = []
    for i in range(fn_train):
        data = [sum(a_all[i, indices_dict[digit]]) for digit in range(class_num)]
        max_index = max(range(len(data)), key=lambda x: data[x])
        max_indices.append(max_index)
    
    print(f'Training accuracy (sum|x|) = {sum(torch.tensor(max_indices) - labels_train == 0) / fn_train}')
    
    # Train neural network classifier
    y0 = labels_train.clone().detach().type(torch.int64).to(device)
    y_hot = torch.nn.functional.one_hot(y0, num_classes=10).float().to(device)
    
    lr = 1e-3
    
    nn_model = torch.nn.Sequential(
        torch.nn.Linear(lca.dict_num, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 10),
    ).to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
    nn_model.zero_grad()
    
    a_all = a_all.to(device)
    
    # Normalize activations
    for i in range(lca.dict_num):
        max_val = torch.max(a_all[i])
        if max_val != 0:
            a_all[i] = a_all[i] / max_val
    
    # Train classifier
    for epoch in range(100):
        y_pred = nn_model(a_all)
        loss = loss_fn(y_pred, y_hot)
        loss.backward()
        if epoch % 10 == 9:
            print('classification loss: {:.2f}'.format(loss.item()))
        optimizer.step()
    
    print("a_all(train): ", a_all.shape)
    print(f'Training accuracy (NN-1000-10) = {sum(torch.max(nn_model(a_all), -1).indices - y0 == 0) / fn_train}') 
    
    # Test the neural network
    print("Testing neural network...")
    yy = labels_test.clone().detach().type(torch.uint8).to(device)
    start_base = time.time()
    
    for i in range(100):
        lca.input = res_model(data_test[i*100:(i+1)*100].to(device))[0].to(device)
        lca.lca_update(20, phi, G, device)
        a = lca.a.clone().detach().type(torch.float)
        if i == 0:
            a_all = a
        else:
            a_all = torch.cat((a_all, a), 0)
    
    end_base = time.time()
    
    # Evaluate test accuracy
    indices = torch.argmax(a_all, dim=1).to('cpu')
    print(f'Testing accuracy (max) = {sum(labels_train[indices] - labels_test == 0) / fn_test}')
    end_max = time.time()
    elapsed_time_max = end_max - start_base
    base_time = end_base - start_base
    print('Elapsed time (max): {:.2f} seconds'.format(elapsed_time_max))
    
    max_indices = []
    start_norm = time.time()
    for i in range(fn_test):
        data = [sum(a_all[i, indices_dict[digit]]) for digit in range(class_num)]
        max_index = max(range(len(data)), key=lambda x: data[x])
        max_indices.append(max_index)
    
    print(f'Testing accuracy (sum|x|) = {sum(torch.tensor(max_indices) - labels_test == 0) / fn_test}')
    end_norm = time.time()
    elapsed_time_norm = end_norm - start_norm
    print('Elapsed time (norm): {:.2f} seconds'.format(elapsed_time_norm + base_time))
    
    start_NN = time.time()
    print(f'Testing accuracy (NN-1000-10) = {sum(torch.max(nn_model(a_all.to(device)), -1).indices - yy == 0) / fn_test}')
    end_NN = time.time()
    elapsed_time_NN = end_NN - start_NN
    print('Elapsed time (NN): {:.2f} seconds'.format(elapsed_time_NN + base_time))
    
    print("PointLCA decoding completed successfully!")


if __name__ == '__main__':
    main()
