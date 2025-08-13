#!/usr/bin/env python3
"""
N-MNIST Training Script
Trains PointNet on pre-processed N-MNIST H5 files.
"""

import os
import sys
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

import datetime
import logging
import provider
import importlib
import argparse

from pathlib import Path
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    parser = argparse.ArgumentParser('N-MNIST training with processed data')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=10, type=int, help='N-MNIST has 10 classes')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number (fixed for processed data)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--data_dir', default='./processed_data/nmnist', help='Directory with processed H5 files')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=10):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        # Handle CUDA/CPU placement
        if not args.use_cpu and torch.cuda.is_available():
            points, target = points.cuda(), target.cuda()

        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


class ProcessedNMNISTDataset(Dataset):
    """N-MNIST dataset loading from processed H5 files"""
    
    def __init__(self, data_dir, split='train', transform=None):
        super(ProcessedNMNISTDataset, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        h5_file = os.path.join(data_dir, f'nmnist_{split}.h5')
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"Processed N-MNIST {split} file not found: {h5_file}")
        
        with h5py.File(h5_file, 'r') as f:
            self.data = f['data'][:]      # Shape: (N, 3, 1024)
            self.labels = f['label'][:]    # Shape: (N,)
        
        print(f"Loaded {split} dataset: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get data and convert to torch tensor
        points = torch.from_numpy(self.data[idx]).float()  # Shape: (3, 1024) - correct for PointNet
        label = int(self.labels[idx])
        
        if self.transform:
            points = self.transform(points)
        
        return points, label


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    # Set random seed for reproducibility
    torch.manual_seed(1234)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir / 'classification'
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir / timestr
    else:
        exp_dir = exp_dir / args.log_dir
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    
    # Create datasets
    train_dataset = ProcessedNMNISTDataset(args.data_dir, 'train')
    test_dataset = ProcessedNMNISTDataset(args.data_dir, 'test')
    
    trainDataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    
    # Handle CUDA/CPU placement
    if not args.use_cpu and torch.cuda.is_available():
        classifier = shutil.cuda()
        criterion = criterion.cuda()
        print("Using CUDA for training")
    else:
        classifier = shutil
        criterion = criterion
        print("Using CPU for training")
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        mean_correct = []
        classifier = classifier.train()
        
        '''learning one epoch'''
        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            # Handle CUDA/CPU placement
            if not args.use_cpu and torch.cuda.is_available():
                points, target = points.cuda(), target.cuda()
            
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            if not args.use_cpu and torch.cuda.is_available():
                points = points.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        scheduler.step()
        mean_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % mean_instance_acc)

        '''Epoch evaluation'''
        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc

            log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    log_string('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
