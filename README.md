# MAPPO Attention Project Repository

Welcome to the repository for our MAPPO (Multi-Agent Proximal Policy Optimization) project. This repository includes the necessary code and scripts to run and manage experiments with MAPPO and modular attention architectures. In addition, we now train our multi-agent systems on three Melting Pot scenario environments:
- **territory__rooms** (number of agents: 9)
- **allelopathic_harvest__open** (number of agents: 16)
- **prisoners_dilemma_in_the_matrix__arena** (number of agents: 8)

## Pretraining and Fine-Tuning

- **Pretraining the Slot Attention Module**  
  We pretrain the Slot Attention module for instance on the `territory__rooms` environment.

- **Fine-Tuning for Multi-Agent RL**  
  We fine-tune the Slot Attention representations for the multi-agent RL task by copying the pretrained model for all agents and then fine-tuning the high-level features using LoRA.

## Environment Setup

### Python Version and Virtual Environment

- **Python Version:** Use **Python 3.10** to ensure compatibility with CUDA 12 for PyTorch.

- **Creating a Conda Environment:**
  ```bash
  conda create -n meltingpot python=3.10
  conda activate meltingpot
  ```

- **Creating a Virtual Environment:**
  ```bash
  python3.10 -m venv meltingpot
  source meltingpot/bin/activate
  ```

### Installing Dependencies

1. **PyTorch Installation (CUDA 12 Compatible):**  
   Install a CUDA 12 compatible version of PyTorch. For example:
   ```bash
   conda install -c nvidia cuda-toolkit=12.1
   conda install conda-forge::jsonnet
   pip install gym[atari,accept-rom-license]
   conda install conda-forge::atari_py 
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   or
   
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

2. **Install mpi4py from GitHub:**
   ```bash
   python -m pip install git+https://github.com/mpi4py/mpi4py
   ```

3. **Installing Flash Attention**  
   This package requires special installation flags and does not exist in the requirements.txt, so it should be installed separately:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

4. **Install Numpy with MKL Optimizations:**  
   For enhanced performance in numerical computations:
   ```bash
   conda install -c conda-forge numpy mkl_fft mkl_random
   ```
   The MKL libraries provide optimized implementations of various math routines, significantly improving performance for linear algebra operations used in deep learning.

5. **Install Required Packages:**  
   Replace the path with your specific requirements file if needed:
   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```

6. **Install this Repository in Editable Mode:**
   ```bash
   pip install -e .
   ```

## Running the Scripts

### Pretraining and Fine-Tuning

- **Pretraining (e.g., for `territory__rooms`):**
  ```bash
  ./run_mappo_territory_rooms_pretrain_slot_att_QSA.sh
  ```

- **Fine-Tuning the Slot Attention Representations:**
  ```bash
  ./run_mappo_territory__room_training_slot_attention_and_rim.sh
  ```
- **Hyperparameter sweeps:** 
   Use the YAML files in `onpolicy/scripts/train/train_meltingpot_scripts/` and from the repo root run `wandb sweep onpolicy/scripts/train/train_meltingpot_scripts/sweep_prisoners_dilemma_hp_search_slot_*.yaml` followed by `wandb agent <entity>/<project>/<sweep_id>`, editing the `hidden_size` and `slot_att_work_path` arguments in the sweep command to match your local directory structure.
