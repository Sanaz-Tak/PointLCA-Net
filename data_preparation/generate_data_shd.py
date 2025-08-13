#!/usr/bin/env python3
"""
SHD Data Generation Script
Processes SHD dataset into H5 format for PointNet training.
"""

import os
import sys
import h5py
import numpy as np
import argparse
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import the working HeidelbergDataset from the reference
sys.path.append(os.path.join(BASE_DIR, '..'))
from heidelberg import HeidelbergDataset

def process_shd(data_dir, output_dir, split='train', num_points=1024):
    """Process SHD dataset to PointNet format"""
    shd_output_dir = os.path.join(output_dir, 'shd')
    os.makedirs(shd_output_dir, exist_ok=True)
    output_file = os.path.join(shd_output_dir, f'shd_{split}.h5')
    
    try:
        if split == 'train':
            data_file = os.path.join(data_dir, 'shd_train.h5')
        else:
            data_file = os.path.join(data_dir, 'shd_test.h5')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"SHD {split} file not found: {data_file}")
        
        dataset = HeidelbergDataset(path=data_file, train=(split == 'train'))
        print(f"Processing {split} split: {len(dataset)} samples")
        
        data_list, labels_list = [], []
        
        for i in range(len(dataset)):
            try:
                processed_data, label = dataset[i]
                
                if hasattr(processed_data, 'cpu'):
                    processed_data = processed_data.cpu().numpy()
                
                if processed_data.shape != (3, 1024):
                    print(f"  Warning: Sample {i} has unexpected shape {processed_data.shape}")
                    continue
                
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
        
    except Exception as e:
        print(f"  Error processing SHD dataset: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate processed SHD training/test data')
    parser.add_argument('--split', choices=['train', 'test', 'both'], 
                       default='both', help='Data split to generate')
    parser.add_argument('--data-dir', default='./data/shd', 
                       help='Directory containing SHD dataset')
    parser.add_argument('--output-dir', default='./processed_data', 
                       help='Output directory for processed data')
    parser.add_argument('--num-points', type=int, default=1024, 
                       help='Number of points per sample (default: 1024)')
    
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    splits = ['train', 'test'] if args.split == 'both' else [args.split]
    
    print("SHD Dataset Processing")
    print("=" * 50)
    print(f"Splits: {', '.join(splits)}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Points per sample: {args.num_points}")
    
    for split in splits:
        print(f"\n--- Processing {split} split ---")
        success = process_shd(args.data_dir, args.output_dir, split, args.num_points)
        if not success:
            print(f"  Failed to process {split} split")
    
    print(f"\n{'='*60}")
    print("SHD data processing completed!")
    print(f"Processed data saved to: {args.output_dir}/shd/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
