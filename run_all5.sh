#!/bin/bash



sbatch --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --mem=16G --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=20:00:00 --mem=24G run10.sh 1

sbatch --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --mem=16G --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=20:00:00 --mem=24G run10.sh 6

sbatch --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --mem=16G --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=20:00:00 --mem=24G run10.sh 5

sbatch --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --mem=16G --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=20:00:00 --mem=24G run10.sh 4

sbatch --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --mem=16G --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=20:00:00 --mem=24G run10.sh 3

sbatch --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --mem=16G --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=20:00:00 --mem=24G run10.sh 2

