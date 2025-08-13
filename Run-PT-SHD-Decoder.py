#!/usr/bin/env python3
"""
SHD PointLCA Decoder
Runs PointLCA algorithm on trained SHD PointNet model.
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import time
import os
import sys
import argparse
import importlib
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))

# No need to import HeidelbergDataset - we'll load processed H5 files directly

def parse_args():
    parser = argparse.ArgumentParser(
        'SHD PointLCA Decoder',
        description='Parametric SHD decoder with configurable LCA and neural network parameters'
    )
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    # Example: --model_path ./log/classification/2025-08-13_08-10/checkpoints/best_model.pth
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained PointNet model checkpoint')
    parser.add_argument('--data_dir', default='./processed_data/shd', 
                       help='Directory with processed H5 files')
    parser.add_argument('--dictionary_size', type=int, default=8156, 
                       help='Size of LCA dictionary (SHD optimal: 8156)')
    parser.add_argument('--neuron_iterations', type=int, default=100, 
                       help='Number of LCA neuron iterations')
    parser.add_argument('--lambda_sparsity', type=float, default=0.2, 
                       help='Sparsity coefficient (lambda)')
    parser.add_argument('--lr_dictionary', type=float, default=0.001, 
                       help='Learning rate for dictionary update')
    parser.add_argument('--lr_neuron', type=float, default=0.001, 
                       help='Learning rate for neuron update')
    parser.add_argument('--batch_size', type=int, default=24, 
                       help='Batch size for LCA processing')
    return parser.parse_args()


class LCA:
    """Locally Competitive Algorithm for sparse coding - based on reference implementation"""
    
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
            # For variable batch sizes, reshape to (batch_size, feature_dim)
            batch_size = self.input.shape[0]
            input = self.input.reshape(batch_size, -1)

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


# ProcessedSHDDataset class removed - using HeidelbergDataset instead



def main():
    args = parse_args()
    
    # Global variables for LCA - will be set after loading data
    class_num = 20
    
    # Set random seed for reproducibility (like reference project)
    torch.manual_seed(1234)
    
    # Set device
    if not args.use_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load processed datasets directly from H5 files (like DVS128 and N-MNIST)
    print("Loading processed datasets...")
    
    # Load train data
    train_file = os.path.join(args.data_dir, 'shd_train.h5')
    with h5py.File(train_file, 'r') as f:
        train_data = torch.from_numpy(f['data'][:]).float()  # Shape: (N, 3, 1024)
        train_labels = torch.from_numpy(f['label'][:]).long()  # Shape: (N,)
    
    # Load test data
    test_file = os.path.join(args.data_dir, 'shd_test.h5')
    with h5py.File(test_file, 'r') as f:
        test_data = torch.from_numpy(f['data'][:]).float()  # Shape: (N, 3, 1024)
        test_labels = torch.from_numpy(f['label'][:]).long()  # Shape: (N,)
    
    print(f"Train data shape: {train_data.shape}, labels: {train_labels.shape}")
    print(f"Test data shape: {test_data.shape}, labels: {test_labels.shape}")
    
    # Set global variables based on actual data size
    global fn_train, fn_test
    fn_train = train_data.shape[0]
    fn_test = test_data.shape[0]
    print(f"Dataset sizes: train={fn_train}, test={fn_test}")
    
    # Set batch size early for use throughout the function
    batch_size = args.batch_size
    
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
            classifier = model_module.get_model(20, normal_channel=False)  # SHD has 20 classes
            
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
    
    # Load data into lca directly from loaded tensors
    lca.data_train = train_data.to(device)  # Already in (batch, channels, points) format
    lca.labels_train = train_labels.to(device)
    
    lca.data_test = test_data.to(device)  # Already in (batch, channels, points) format
    lca.labels_test = test_labels.to(device)
    
    print("dataset_train shape:", lca.data_train.shape, lca.data_train.device)
    print("labels_train shape:", len(lca.labels_train))
    
    # Extract all feature maps for dictionary initialization
    print("Extracting feature maps for dictionary initialization...")
    all_feature_maps = torch.tensor([]).to(device)
    
    # Use configurable batch size for feature extraction
    num_batches = (lca.dict_num + batch_size - 1) // batch_size  # Ceiling division
    for i in range(num_batches):
        with torch.no_grad():
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, lca.dict_num)  # Use dict_num instead of fn_train
            feature_maps = res_model(lca.data_train[start_idx:end_idx])[0].to(device)
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
    # Process full dataset like reference project (was: min(100, fn_train // batch_size))
    batch_size = args.batch_size
    train_iterations = fn_train // batch_size  # Process full dataset like reference project
    print(f"Training for {train_iterations} iterations with batches of {batch_size} (full dataset)")
    print(f"Will process {train_iterations * batch_size} + {fn_train % batch_size} = {fn_train} total training samples")
    
    # Track which samples were processed
    processed_indices = []
    
    for i in range(train_iterations):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        processed_indices.extend(range(start_idx, end_idx))
        
        lca.input = res_model(lca.data_train[start_idx:end_idx])[0].to(device)
        lca.lca_update(20, phi, G)
        a = lca.a.clone().detach().type(torch.float).to('cpu')
        if i == 0:
            a_all = a
        else:
            a_all = torch.cat((a_all, a), 0)
    
    # Handle remaining training samples if batch_size doesn't divide evenly
    if fn_train % batch_size != 0:
        start_idx = train_iterations * batch_size
        processed_indices.extend(range(start_idx, fn_train))
        
        lca.input = res_model(lca.data_train[start_idx:])[0].to(device)
        lca.lca_update(20, phi, G)
        a = lca.a.clone().detach().type(torch.float).to('cpu')
        a_all = torch.cat((a_all, a), 0)
    
    print("a_all(train): ", a_all.shape)
    print(f"Processed {len(processed_indices)} samples: {processed_indices[:10]}...")
    
    # Evaluate training accuracy only on processed samples
    num_processed = a_all.shape[0]
    indices = torch.argmax(a_all, dim=1).to('cpu')  # Get the indices of the maximum values along dimension 1
    processed_labels = lca.labels_train[processed_indices]
    
    # Fix: indices are dictionary indices (0-999), not sample indices
    # We need to compare the predicted class (from argmax) with the actual labels
    predicted_classes = indices  # These are the predicted class indices (0-19 for 20 classes)
    actual_labels = processed_labels
    
    # Calculate accuracy by comparing predicted vs actual
    correct_predictions = (predicted_classes == actual_labels).sum().item()
    accuracy = correct_predictions / num_processed
    print(f'Training accuracy (max) = {accuracy:.4f} ({correct_predictions}/{num_processed})')
    
    # Calculate indices_dict more safely
    indices_dict = {}
    for digit in range(class_num):
        digit_indices = torch.nonzero(processed_labels == digit).squeeze()
        if digit_indices.numel() > 0:
            # Ensure it's a 1D tensor with correct dtype for indexing
            if digit_indices.dim() == 0:
                indices_dict[digit] = digit_indices.unsqueeze(0).long()
            else:
                indices_dict[digit] = digit_indices.long()
        else:
            indices_dict[digit] = torch.tensor([], dtype=torch.long)
    
    # Calculate sum|x| accuracy more safely
    max_indices = []
    for i in range(num_processed):
        data = []
        for digit in range(class_num):
            if digit in indices_dict and indices_dict[digit].numel() > 0:
                try:
                    data.append(sum(a_all[i, indices_dict[digit]]))
                except:
                    data.append(0.0)
            else:
                data.append(0.0)
        max_index = max(range(len(data)), key=lambda x: data[x])
        max_indices.append(max_index)
    
    print(f'Training accuracy (sum|x|) = {sum(torch.tensor(max_indices) - processed_labels == 0) / num_processed}')
    
    # Train neural network classifier on processed samples
    y0 = processed_labels.clone().detach().type(torch.int64).to(device)
    y_hot = torch.nn.functional.one_hot(y0, num_classes=class_num).float().to(device)
    
    lr = 1e-3  # Hardcoded like other decoders
    
    # Configurable neural network architecture
    hidden_size = 1000
    nn_model = torch.nn.Sequential(
        torch.nn.Linear(lca.dict_num, hidden_size),
        #torch.nn.Dropout(p=0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, class_num),
        #torch.nn.Softmax(dim=1)
    ).to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
    nn_model.zero_grad()
    
    a_all = a_all.to(device)
    
    # Normalize activations like the original
    for i in range(a_all.shape[0]):  # Use actual processed data size
        max_val = torch.max(a_all[i])
        if max_val != 0:
            a_all[i] = a_all[i] / max_val
        else:
            pass
    
    # Train classifier
    for epoch in range(100):  # Hardcoded like other decoders
        y_pred = nn_model(a_all)
        loss = loss_fn(y_pred, y_hot)
        loss.backward()
        if epoch % 10 == 9:
            print('classification loss: {:.2f}'.format(loss.item()))
        optimizer.step()
    
    print("a_all(train): ", a_all.shape)
    print(f'Training accuracy (NN-1000-20) = {sum(torch.max(nn_model(a_all), -1).indices - y0 == 0) / num_processed}')
    
    # Test the neural network
    print("Testing neural network...")
    yy = lca.labels_test.clone().detach().type(torch.uint8).to(device)
    start_base = time.time()
    
    # Test the neural network - process all test data like N-MNIST and DVS128
    print("Testing neural network...")
    yy = lca.labels_test.clone().detach().type(torch.uint8).to(device)
    start_base = time.time()
    
    # Process all test data in batches like the working decoders
    num_test_batches = fn_test // batch_size
    for i in range(num_test_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        lca.input = res_model(lca.data_test[start_idx:end_idx])[0].to(device)
        lca.lca_update(20, phi, G)
        a = lca.a.clone().detach().type(torch.float)
        if i == 0:
            a_all = a
        else:
            a_all = torch.cat((a_all, a), 0)
    
    # Handle remaining samples if batch_size doesn't divide evenly
    if fn_test % batch_size != 0:
        start_idx = num_test_batches * batch_size
        lca.input = res_model(lca.data_test[start_idx:])[0].to(device)
        lca.lca_update(20, phi, G)
        a = lca.a.clone().detach().type(torch.float)
        a_all = torch.cat((a_all, a), 0)
    
    end_base = time.time()
    
    # Evaluate test accuracy using full dataset like working decoders
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
    print(f'Testing accuracy (NN-1000-20) = {sum(torch.max(nn_model(a_all.to(device)), -1).indices - yy == 0) / fn_test}')
    end_NN = time.time()
    elapsed_time_NN = end_NN - start_NN
    print('Elapsed time (NN): {:.2f} seconds'.format(elapsed_time_NN + base_time))
    
    print("PointLCA decoding completed successfully!")


if __name__ == '__main__':
    main()
