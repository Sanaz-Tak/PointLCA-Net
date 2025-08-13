#!/usr/bin/env python3
"""
DVS128 PointLCA Decoder
Runs PointLCA algorithm on trained DVS128 PointNet model.
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
    parser = argparse.ArgumentParser('DVS128 PointLCA Decoder')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    # Example: --model_path ./log/classification/2025-08-13_08-10/checkpoints/best_model.pth
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained PointNet model checkpoint')
    parser.add_argument('--data_dir', default='./processed_data/dvs128', 
                       help='Directory with processed H5 files')
    parser.add_argument('--dictionary_size', type=int, default=28606, 
                       help='Size of LCA dictionary (DVS128 optimal: 28606)')
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

    def lca_update(self, n, phi, G):
        # Get device from phi tensor
        device = phi.device
        u_list = [torch.zeros([1, self.dict_num]).to(device)]
        a_list = [self.threshold(u_list[0], 'soft', True, self.landa).to(device)]

        if n == 0:
            dict = self.dictionary.reshape(self.dict_num, -1)
            phi = dict.T
            phi = phi.to(device)
            I = torch.eye(self.dict_num).to(device)
            G = torch.mm(phi.T, phi) - I
            G = phi.to(device)
            input = self.input.detach().reshape(-1)
        elif n == 1:
            input = self.input.reshape(fn_train, -1)
        elif n == 2:
            input = self.input.reshape(fn_test, -1)
        else:
            # Calculate the actual number of samples we can fit
            total_size = self.input.numel()
            if total_size % 1024 == 0:
                num_samples = total_size // 1024
                input = self.input.reshape(num_samples, 1024)
            else:
                # Fallback: just flatten
                input = self.input.reshape(-1)

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


class ProcessedDVS128Dataset:
    """Dataset class for processed DVS128 H5 files"""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        
        # Find the processed H5 file in the DVS128 directory structure
        # DVS128 uses a specific folder structure: C11_TS1_1024/W05S0125/
        base_dir = os.path.join(data_dir, 'C11_TS1_1024', 'W05S0125')
        h5_file = os.path.join(base_dir, f'{split}_0.h5')
        
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"Processed DVS128 {split} file not found: {h5_file}")
        
        with h5py.File(h5_file, 'r') as f:
            self.data = f['data'][:]      # Shape: (N, 1, 1024, 3) - DVS128 format
            self.labels = f['label'][:]    # Shape: (N, 1)
        
        # Reshape data to match our expected format
        # DVS128 data is (N, 1, 1024, 3), we want (N, 3, 1024)
        self.data = self.data.squeeze(1)  # Remove middle dimension: (N, 1024, 3)
        self.data = self.data.transpose(0, 2, 1)  # (N, 3, 1024)
        
        # Labels are (N, 1), squeeze to (N,)
        self.labels = self.labels.squeeze(1)
        
        print(f"Loaded {split} dataset: {len(self.data)} samples, shape: {self.data.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), int(self.labels[idx])


def extract_features(model, data_loader, device, max_samples=1000):
    """Extract features from trained PointNet model"""
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for i, (points, target) in enumerate(data_loader):
            if i >= max_samples:
                break
                
            points = points.to(device)
            # PointNet expects (batch_size, 3, num_points)
            if points.dim() == 2:
                points = points.unsqueeze(0)  # Add batch dimension
            
            # Extract features (assuming the model has a feature extraction method)
            try:
                # Try to get features from the model
                if hasattr(model, 'feat'):
                    feat = model.feat(points)[0]  # Get features from feature extractor
                else:
                    # If no feature extractor, use the last layer before classification
                    feat = model(points)[0]  # Get features from forward pass
                
                features.append(feat.cpu())
                labels.append(target)
                
            except Exception as e:
                print(f"Error extracting features from sample {i}: {e}")
                continue
    
    if not features:
        raise RuntimeError("No features extracted successfully")
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    print(f"Extracted features shape: {features.shape}")
    return features, labels


def main():
    args = parse_args()
    
    # Global variables for LCA
    global fn_train, fn_test
    fn_train = 4180  # DVS128 train size
    fn_test = 14667  # DVS128 test size
    class_num = 11
    
    # Set random seed for reproducibility (like reference project)
    torch.manual_seed(1234)
    
    # Set device
    if not args.use_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = ProcessedDVS128Dataset(args.data_dir, 'train')
    test_dataset = ProcessedDVS128Dataset(args.data_dir, 'test')
    
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
            classifier = model_module.get_model(11, normal_channel=False)  # DVS128 has 11 classes
            
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
    
    # Initialize LCA first
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
    
    # Load data into lca like the original working version
    lca.data_train = torch.zeros([fn_train, 3, lca.feature_size])
    lca.labels_train = torch.zeros(fn_train, dtype=torch.int)
    
    lca.data_test = torch.zeros([fn_test, 3, lca.feature_size])
    lca.labels_test = torch.zeros(fn_test, dtype=torch.int)
    
    dataiter = iter(train_loader)
    lca.data_train, lca.labels_train = next(dataiter)
    # PointNet expects (batch, channels, points) format
    # Data is already in (batch, channels, points) format, no need to permute
    lca.data_train = lca.data_train.to(device)
    
    dataiter = iter(test_loader)
    lca.data_test, lca.labels_test = next(dataiter)
    # PointNet expects (batch, channels, points) format
    # Data is already in (batch, channels, points) format, no need to permute
    lca.data_test = lca.data_test.to(device)
    
    print("dataset_train shape:", lca.data_train.shape, lca.data_train.device)
    print("labels_train shape:", len(lca.labels_train))
    
    # Extract all feature maps for dictionary initialization
    print("Extracting feature maps for dictionary initialization...")
    all_feature_maps = torch.tensor([]).to(device)
    
    # Use the original working approach: process in batches of 2
    for i in range(lca.dict_num//2):
        with torch.no_grad():
            feature_maps = res_model(lca.data_train[i*2:(i+1)*2])[0].to(device)
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
    # Use the original working approach: 14303 iterations with batches of 2
    for i in range(14303):
        lca.input = res_model(lca.data_train[i*2:(i+1)*2])[0].to(device)
        lca.lca_update(20, phi, G)
        a = lca.a.clone().detach().type(torch.float).to('cpu')
        if i == 0:
            a_all = a
        else:
            a_all = torch.cat((a_all, a), 0)
    
    print("a_all(train): ", a_all.shape)
    
    # Evaluate training accuracy
    indices = torch.argmax(a_all, dim=1).to('cpu')  # Get the indices of the maximum values along dimension 1
    print(f'Training accuracy (max) = {sum(lca.labels_train[indices] - lca.labels_train == 0) / fn_train}')
    
    indices_dict = {}
    for digit in range(class_num):
        indices_dict[digit] = torch.nonzero(lca.labels_train[0:lca.dict_num] == digit).squeeze()
    
    max_indices = []
    for i in range(fn_train):
        data = [sum(a_all[i, indices_dict[digit]]) for digit in range(class_num)]
        max_index = max(range(len(data)), key=lambda x: data[x])
        max_indices.append(max_index)
    
    print(f'Training accuracy (sum|x|) = {sum(torch.tensor(max_indices) - lca.labels_train == 0) / fn_train}')
    
        # Train neural network classifier
    y0 = lca.labels_train.clone().detach().type(torch.int64).to(device)
    y_hot = torch.nn.functional.one_hot(y0, num_classes=11).float().to(device)
    
    lr = 1e-3
    
    nn_model = torch.nn.Sequential(
        torch.nn.Linear(lca.dict_num, 1000),
        #torch.nn.Dropout(p=0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 11),
        #torch.nn.Softmax(dim=1)
    ).to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
    nn_model.zero_grad()
    
    a_all = a_all.to(device)
    
    # Normalize activations like the original
    for i in range(lca.dict_num):
        max_val = torch.max(a_all[i])
        if max_val != 0:
            a_all[i] = a_all[i] / max_val
        else:
            pass
    
    # Train classifier
    for epoch in range(100):
        y_pred = nn_model(a_all)
        loss = loss_fn(y_pred, y_hot)
        loss.backward()
        if epoch % 10 == 9:
            print('classification loss: {:.2f}'.format(loss.item()))
        optimizer.step()
    
    print("a_all(train): ", a_all.shape)
    print(f'Training accuracy (NN-1000-11) = {sum(torch.max(nn_model(a_all), -1).indices - y0 == 0) / fn_train}')
    
    # Test the neural network
    print("Testing neural network...")
    yy = lca.labels_test.clone().detach().type(torch.uint8).to(device)
    start_base = time.time()
    
    # Use the original working approach: process all test data in batches of 2
    num_test_batches = fn_test // 2
    for i in range(num_test_batches):
        lca.input = res_model(lca.data_test[i*2:(i+1)*2])[0].to(device)
        lca.lca_update(20, phi, G)
        a = lca.a.clone().detach().type(torch.float)
        if i == 0:
            a_all = a
        else:
            a_all = torch.cat((a_all, a), 0)
    
    # Handle odd number of samples
    if fn_test % 2 != 0:
        lca.input = res_model(lca.data_test[-1:])[0].to(device)
        lca.lca_update(20, phi, G)
        a = lca.a.clone().detach().type(torch.float)
        a_all = torch.cat((a_all, a), 0)
    
    end_base = time.time()
    
    # Evaluate test accuracy
    indices = torch.argmax(a_all, dim=1).to('cpu')  # Get the indices of the maximum values along dimension 1
    print(f'Testing accuracy (max) = {sum(lca.labels_train[indices] - lca.labels_test == 0) / fn_test}')
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
    
    print(f'Testing accuracy (sum|x|) = {sum(torch.tensor(max_indices) - lca.labels_test == 0) / fn_test}')
    end_norm = time.time()
    elapsed_time_norm = end_norm - start_norm
    print('Elapsed time (norm): {:.2f} seconds'.format(elapsed_time_norm + base_time))
    
    start_NN = time.time()
    print(f'Testing accuracy (NN-1000-11) = {sum(torch.max(nn_model(a_all.to(device)), -1).indices - yy == 0) / fn_test}')
    end_NN = time.time()
    elapsed_time_NN = end_NN - start_NN
    print('Elapsed time (NN): {:.2f} seconds'.format(elapsed_time_NN + base_time))
    
    print("PointLCA decoding completed successfully!")


if __name__ == '__main__':
    main()
