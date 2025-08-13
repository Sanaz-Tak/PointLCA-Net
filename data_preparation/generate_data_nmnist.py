#!/usr/bin/env python3
"""
N-MNIST Data Generation Script
Processes N-MNIST binary spike files into H5 format for PointNet training.
"""

import os
import sys
import h5py
import numpy as np
import argparse
import glob
import zipfile
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

class NMNISTDataset:
    """N-MNIST dataset class for loading and processing binary spike files"""
    
    def __init__(
        self, path='./data/nmnist',
        train=True,
        sampling_time=1, 
        sample_length=300,
        transform=None, 
        download=True,
    ):
        self.path = path
        if train:
            data_path = path + '/Train'
            source = 'https://www.dropbox.com/sh/tg2ljlbmtzygrag/'\
                'AABlMOuR15ugeOxMCX0Pvoxga/Train.zip'
        else:
            data_path = path + '/Test'
            source = 'https://www.dropbox.com/sh/tg2ljlbmtzygrag/'\
                'AADSKgJ2CjaBWh75HnTNZyhca/Test.zip'

        if download is True:
            # N-MNIST dataset citation: Orchard et al., Frontiers in Neuroscience, 2015

            if len(glob.glob(f'{data_path}/')) == 0:  # dataset does not exist
                print(f'Downloading N-MNIST {"training" if train else "testing"} dataset...')
                os.system(f'wget {source} -P {self.path}/ -q --show-progress')
                with zipfile.ZipFile(data_path + '.zip') as zip_file:
                    for member in zip_file.namelist():
                        zip_file.extract(member, self.path)
                print('Download complete.')
        else:
            assert len(glob.glob(f'{data_path}/')) > 0, \
                f'Dataset does not exist. Either set download=True '\
                f'or download it from '\
                f'https://www.garrickorchard.com/datasets/n-mnist '\
                f'to {data_path}/'

        self.samples = glob.glob(f'{data_path}/*/*.bin')
        self.sampling_time = sampling_time
        self.num_time_bins = int(sample_length/sampling_time)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def get_sample(self, i):
        """Get a single sample with full processing"""
        filename = self.samples[i]
        label = int(filename.split('/')[-2])
        processed_data = self.process_binary_file(filename)
        return processed_data, label

    def process_binary_file(self, filename):
        """Process a single binary file with complete training script logic"""
        # Read binary file using the same function as training script
        x_event, y_event, c_event, t_event = read_2d_spikes(filename)
        
        # Apply the same processing as training script
        x = x_event.astype(np.float32)
        x = x / (np.max(x) + 1e-8)
        
        y = y_event.astype(np.float32)
        y = y / (np.max(y) + 1e-8)
        
        t = t_event.astype(np.float32)
        t = t / (np.max(t) + 1e-8)
        
        # Create 3-channel tensor [x, y, t] (same as training script)
        ev = np.column_stack([x, y, t]).T  # Shape: (3, num_events)
        
        # Dynamic windowing and sampling (same as training script)
        data_size = ev.shape[1]
        
        if data_size >= 4096:
            window_size = data_size // 8
            windows = [ev[:, i * window_size: (i + 1) * window_size] for i in range(8)]
            nn = 128
        elif data_size >= 2048:
            window_size = data_size // 4
            windows = [ev[:, i * window_size: (i + 1) * window_size] for i in range(4)]
            nn = 256
        elif data_size >= 1024:
            window_size = data_size // 2
            windows = [ev[:, i * window_size: (i + 1) * window_size] for i in range(2)]
            nn = 512
        else:
            # Pad to 1024 if not enough points
            if data_size < 1024:
                padding = np.tile(ev[:, -1:], (1, 1024 - data_size))
                ev = np.hstack([ev, padding])
            windows = [ev]
            nn = 1024
        
        # Random sampling within each window
        sampled_parts = []
        for window in windows:
            if window.shape[1] >= nn:
                # Randomly sample nn points
                indices = np.random.choice(window.shape[1], nn, replace=False)
                sampled_parts.append(window[:, indices])
            else:
                # If window is smaller, pad it
                padding = np.tile(window[:, -1:], (1, nn - window.shape[1]))
                sampled_parts.append(np.hstack([window, padding]))
        
        # Concatenate all windows
        ev_all = np.hstack(sampled_parts)
        
        # Final random sampling to get exactly 1024 points
        if ev_all.shape[1] >= 1024:
            indices = np.random.choice(ev_all.shape[1], 1024, replace=False)
            ev_all = ev_all[:, indices]
        else:
            # Pad to 1024 if still not enough
            padding = np.tile(ev_all[:, -1:], (1, 1024 - ev_all.shape[1]))
            ev_all = np.hstack([ev_all, padding])
        
        # Return as (3, 1024) tensor ready for training
        return ev_all.astype(np.float32)

def read_2d_spikes(filename):
    """Reads two dimensional binary spike file and returns event data.
    Same format used in neuromorphic datasets NMNIST & NCALTECH101.
    
    Binary file encoding:
    - Each spike event: 40 bit number
    - First 8 bits (bits 39-32): xID of neuron
    - Next 8 bits (bits 31-24): yID of neuron  
    - Bit 23: sign of spike (0=OFF, 1=ON)
    - Last 23 bits (bits 22-0): timestamp in microseconds
    """
    with open(filename, 'rb') as input_file:
        input_byte_array = input_file.read()
    
    input_as_int = np.asarray([x for x in input_byte_array])
    x_event = input_as_int[0::5]
    y_event = input_as_int[1::5]
    c_event = input_as_int[2::5] >> 7
    t_event = (
        (input_as_int[2::5] << 16)
        | (input_as_int[3::5] << 8)
        | (input_as_int[4::5])
    ) & 0x7FFFFF
    
    # Convert spike times to ms
    return x_event, y_event, c_event, t_event

def process_nmnist(data_dir, output_dir, split='train', num_points=1024):
    """Process N-MNIST dataset and save to H5 format"""
    nmnist_output_dir = os.path.join(output_dir, 'nmnist')
    os.makedirs(nmnist_output_dir, exist_ok=True)
    output_file = os.path.join(nmnist_output_dir, f'nmnist_{split}.h5')
    
    dataset = NMNISTDataset(path=data_dir, train=(split == 'train'), download=True)
    print(f"Processing {split} split: {len(dataset)} samples")
    
    data_list, labels_list = [], []
    
    for i in range(len(dataset)):
        try:
            processed_data, label = dataset.get_sample(i)
            data_list.append(processed_data)
            labels_list.append(label)
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} samples")
                
        except Exception as e:
            print(f"  Warning: Failed to process sample {i}: {e}")
            continue
    
    if not data_list:
        print(f"  Error: No data generated for {split}")
        return None
    
    data_array = np.array(data_list, dtype=np.float32)
    labels_array = np.array(labels_list, dtype=np.int32)
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('data', data=data_array)
        f.create_dataset('label', data=labels_array)
    
    print(f"  Saved {len(data_array)} samples to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate processed N-MNIST training/test data')
    parser.add_argument('--split', choices=['train', 'test', 'both'], 
                       default='both', help='Data split to generate')
    parser.add_argument('--data-dir', default='./data/nmnist', 
                       help='Directory containing N-MNIST dataset')
    parser.add_argument('--output-dir', default='./processed_data', 
                       help='Output directory for processed data')
    parser.add_argument('--num-points', type=int, default=1024, 
                       help='Number of points per sample (default: 1024)')
    parser.add_argument('--download', action='store_true', default=True,
                       help='Download dataset if not available (default: True)')
    
    args = parser.parse_args()
    
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    splits = ['train', 'test'] if args.split == 'both' else [args.split]
    
    print("N-MNIST Data Processing")
    print(f"Processing splits: {', '.join(splits)}")
    
    for split in splits:
        success = process_nmnist(args.data_dir, args.output_dir, split, args.num_points)
        if not success:
            print(f"Failed to process {split} split")
    
    print("N-MNIST data processing completed.")

if __name__ == "__main__":
    main()
