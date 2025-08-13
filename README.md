# Neuromorphic PointNet: Event-Based Point Cloud Classification

A specialized implementation of PointNet and PointNet++ architectures for neuromorphic datasets, featuring end-to-end pipelines for N-MNIST, DVS128, and SHD datasets with PointLCA decoding capabilities.

## Project Overview

This project extends the original PointNet/PointNet++ architectures to process neuromorphic event-based data, converting temporal spike events into 3D point clouds for deep learning classification. It includes complete data processing pipelines, training scripts, and PointLCA (Locally Competitive Algorithm) decoding for sparse feature extraction.


## Supported Datasets

| Dataset | Classes | Description | Input Format | Output Format |
|--|--|--|--|--|
| **N-MNIST** | 10 | Neuromorphic MNIST digits from saccadic eye movements | Binary spike files | H5 (N, 3, 1024) |
| **DVS128** | 11 | Dynamic Vision Sensor gesture recognition | .aedat files | H5 (N, 3, 1024) |
| **SHD** | 20 | Spiking Heidelberg Digits audio classification | H5 files | H5 (N, 3, 1024) |

## Architecture

```
Event Data → Point Cloud Conversion → PointNet/PointNet++ → PointLCA Decoder
    ↓              ↓                      ↓                    ↓
Raw Events → 3D Coordinates → Feature Extraction → Sparse Representation
```

## Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.6+
- CUDA 10.1+ (for GPU acceleration)
- H5Py for data storage

### Setup
```bash
# Clone the repository
git clone https://github.com/Sanaz-Tak/PointLCA-Net.git
cd PointLCA-Net

# Install dependencies
pip install torch torchvision h5py numpy matplotlib tqdm

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## Quick Start

**Note**: All training scripts require you to specify a `--log_dir` argument for your experiment. This creates a unique directory for your training logs, checkpoints, and results.

### 1. Data Preparation

#### N-MNIST Dataset
```bash
# Download and process N-MNIST data
python data_preparation/generate_data_nmnist.py --split both --num-points 1024

# Data will be saved to: ./processed_data/nmnist/
```

#### DVS128 Dataset
```bash
# Process DVS128 .aedat files
python data_preparation/generate_data_dvs128.py --split both --num-points 1024

# Data will be saved to: ./processed_data/dvs128/
```

#### SHD Dataset
```bash
# Process SHD dataset
python data_preparation/generate_data_shd.py --split both --num-points 1024

# Data will be saved to: ./processed_data/shd/
```

### 2. Training PointNet Models

#### N-MNIST Training
```bash
# Train PointNet on N-MNIST
python train_classification_nmnist_processed.py \
    --batch_size 24 \
    --epoch 200 \
    --num_point 1024 \
    --learning_rate 0.001 \
    --log_dir nmnist_experiment

# Monitor training in: ./log/classification/nmnist_experiment/
```

#### DVS128 Training
```bash
# Train PointNet on DVS128
python train_classification_dvs128_processed.py \
    --batch_size 24 \
    --epoch 200 \
    --num_point 1024 \
    --learning_rate 0.001 \
    --log_dir dvs128_experiment

# Monitor training in: ./log/classification/dvs128_experiment/
```

#### SHD Training
```bash
# Train PointNet on SHD
python train_classification_shd_processed.py \
    --batch_size 24 \
    --epoch 200 \
    --num_point 1024 \
    --learning_rate 0.001 \
    --log_dir shd_experiment

# Monitor training in: ./log/classification/shd_experiment/
```

### 3. PointLCA Decoding

#### N-MNIST Decoding
```bash
# Run PointLCA decoder on N-MNIST
python Run-PT-NMNIST-Decoder.py \
    --model_path ./log/classification/nmnist_experiment/checkpoints/best_model.pth \
    --data_dir ./processed_data/nmnist \
    --num_point 1024
```

#### DVS128 Decoding
```bash
# Run PointLCA decoder on DVS128
python Run-PT-DVS128-Decoder.py \
    --model_path ./log/classification/dvs128_experiment/checkpoints/best_model.pth \
    --data_dir ./processed_data/dvs128 \
    --num_point 1024
```

#### SHD Decoding
```bash
# Run PointLCA decoder on SHD
python Run-PT-SHD-Decoder.py \
    --model_path ./log/classification/shd_experiment/checkpoints/best_model.pth \
    --data_dir ./processed_data/shd \
    --num_point 1024
```

## Configuration

### Key Hyperparameters

| Parameter | N-MNIST | DVS128 | SHD | Description |
|--|--|--|--|--|
| `dictionary_size` | 60000 | 28606 | 8156 | LCA dictionary size |
| `lambda_sparsity` | 0.2 | 0.2 | 0.2 | Sparsity coefficient |
| `batch_size` | 24 | 24 | 24 | Processing batch size |
| `neuron_iterations` | 100 | 100 | 100 | LCA neuron update iterations |
| `num_points` | 1024 | 1024 | 1024 | Points per point cloud |

### Training Parameters

| Parameter | Default | Description |
|--|--|--|
| `--epoch` | 200 | Number of training epochs |
| `--learning_rate` | 0.001 | Learning rate for optimization |
| `--batch_size` | 24 | Training batch size |
| `--num_point` | 1024 | Number of points per sample |
| `--use_cpu` | False | Force CPU-only training |
| `--gpu` | 0 | GPU device ID |

## Project Structure

```
PointLCA-Net/
├── data_preparation/           # Data processing scripts
│   ├── generate_data_nmnist.py    # N-MNIST data generation
│   ├── generate_data_dvs128.py    # DVS128 data generation
│   ├── generate_data_shd.py       # SHD data generation
│   └── PyAedatTools/              # DVS128 processing utilities
├── models/                    # PointNet/PointNet++ architectures
│   ├── pointnet_cls.py           # PointNet classification
│   └── pointnet2_cls_ssg.py     # PointNet++ classification
├── Run-PT-*-Decoder.py        # PointLCA decoding scripts
├── train_classification_*.py  # Training scripts
├── provider.py                # Data augmentation utilities
├── heidelberg.py              # SHD dataset loader
└── README.md    # This file
```

## Contributing

We welcome contributions to improve the neuromorphic PointNet implementation:

1. **Bug Reports**: Open issues for any problems you encounter
2. **Feature Requests**: Suggest new datasets or functionality
3. **Code Contributions**: Submit pull requests with improvements
4. **Documentation**: Help improve documentation and examples

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/Sanaz-Tak/PointLCA-Net.git
cd PointLCA-Net

# Create a development branch
git checkout -b feature/new-dataset

# Make your changes and test
python -m pytest tests/

# Submit a pull request
```

## References

### Original PointNet Implementation
- **PointNet**: Qi et al., "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation", CVPR 2017
- **PointNet++**: Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space", NeurIPS 2017

### Neuromorphic Datasets
- **N-MNIST**: Orchard et al., "Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades", Frontiers in Neuroscience, 2015
- **DVS128**: Amir et al., "A Low Power, Fully Event-Based Gesture Recognition System", CVPR, 2017
- **SHD**: Cramer et al., "The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks", IEEE TNNLS, 2020

### PointLCA Algorithm
- **LCA**: Rozell et al., "Locally Competitive Algorithms for Sparse Approximation", IEEE TSP, 2008

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

### Original PointNet Implementation
This project builds upon the excellent PyTorch implementation of PointNet and PointNet++ by [yanx27](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). Their work provided the foundation for the models used in this neuromorphic extension.


## Contact

For questions, issues, or collaboration opportunities:

- **Issues**: [GitHub Issues](https://github.com/Sanaz-Tak/PointLCA-Net/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Sanaz-Tak/PointLCA-Net/discussions)
---

**Note**: This project extends the original PointNet/PointNet++ architectures for neuromorphic computing applications. While we acknowledge and build upon the excellent work of the original authors, this implementation focuses specifically on event-based data processing and PointLCA decoding capabilities.
