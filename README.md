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

- **Python Version:**  
  Use **Python 3.10** to ensure compatibility with CUDA 12 for PyTorch.

- **Creating a Conda Environment:**  
  ```bash
  conda create -n mappo_project python=3.10
  conda activate mappo_project
  ```

- **Creating a Virtual Environment:**  
  ```bash
  python3.10 -m venv mappo_project
  source mappo_project/bin/activate
  ```

### Installing Dependencies

1. **PyTorch Installation (CUDA 12 Compatible):**  
   Install a CUDA 12 compatible version of PyTorch. For example:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120
   ```

2. **Install mpi4py from GitHub:**  
   ```bash
   python -m pip install git+https://github.com/mpi4py/mpi4py
   ```

3. **Install Required Packages:**  
   Replace the path with your specific requirements file if needed:
   ```bash
   pip install --no-cache-dir -r ~/projects/def-irina/memole/LSTM/requirements.txt
   ```

4. **Install this Repository in Editable Mode:**  
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
