#!/bin/bash



sbatch --reservation=ubuntu2204 --nodes=1  --ntasks-per-node=1 --cpus-per-task=9 --gpus-per-task=1 --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=50:00:00 --mem=24G run10.sh 67

#sbatch --reservation=ubuntu2204 --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --gpus-per-task=1 --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=50:00:00 --mem=24G run10.sh 26

#sbatch --reservation=ubuntu2204 --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --gpus-per-task=1 --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=50:00:00 --mem=24G run10.sh 27

#sbatch --reservation=ubuntu1804 --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --mem=16G --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=20:00:00 --mem=24G run10.sh 5

#sbatch --reservation=ubuntu1804 --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --mem=16G --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=20:00:00 --mem=24G run10.sh 4

#sbatch --reservation=ubuntu1804 --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --mem=16G --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=20:00:00 --mem=24G run10.sh 3

#sbatch --reservation=ubuntu1804 --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --mem=16G --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=20:00:00 --mem=24G run10.sh 2
