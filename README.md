# ModBERT: a DNA modification and genetic variant-aware DNA large language model


ModBERT is a DNA language model built on a five-letter probabilistic genome representation in which each position is a distribution over A, C, G, T, M (5mC).

## Environment Setup

### Requirements

- Python 3.12
- Linux x86_64
- NVIDIA GPU with a CUDA 12 compatible driver (for GPU-accelerated PyTorch)

The pinned dependencies include CUDA 12 libraries and `triton`, which are only available for Linux x86_64. They will not install on macOS or Windows. CPU-only use requires a different PyTorch build (see the troubleshooting section below).

### Installation

1. Create and activate a conda environment:

   ```bash
   conda create -n modbert python=3.12 -y
   conda activate modbert
   ```

2. Install the pinned dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Verify the installation:

   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

   The output should show `2.5.1` and `True` on a machine with a working GPU.

### Troubleshooting

- **`No matching distribution found` for `nvidia-*` or `triton`:** You are on a platform these packages don't support. Use a Linux x86_64 machine, or remove the `nvidia-*` and `triton` lines and install PyTorch separately from the [PyTorch install page](https://pytorch.org/get-started/locally/) for your platform.
