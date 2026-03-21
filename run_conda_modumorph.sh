#!/usr/bin/env bash
#SBATCH --mail-user=sviswasam@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu:L40S:1
#SBATCH -t 23:59:00
#SBATCH --mem 64G
#SBATCH --job-name="train_isaac"

#SBATCH --output=/home/sviswasam/dr/ModuMorph/logs/output_walk_isaac.log
#SBATCH --error=/home/sviswasam/dr/ModuMorph/logs/err_walk_isaac.err

# --- ISAACGYM CONFIG ---

# 0. Initialize module system
source /etc/profile.d/modules.sh

# 1. Load only CUDA — Isaac Gym does not need LLVM, GLEW, or libX11
module load cuda/11.1.0/pdhgaya

# 2. Force GCC for gymtorch JIT compilation
#    Isaac Gym's gymtorch.py compiles a C++ extension at first import.
#    The cluster's clang (llvm-18) conflicts with PyTorch's LLVM — use GCC instead.
export CC=gcc
export CXX=g++

# 3. Library paths — only what Isaac Gym actually needs
#    conda env lib first so libstdc++ is new enough for Isaac Gym binaries
export LD_LIBRARY_PATH=/home/sviswasam/miniforge3/envs/ig_train/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sviswasam/.mujoco/mujoco210/bin

# 4. Explicitly unset anything that could cause LLVM/OpenGL conflicts
unset LD_PRELOAD
unset LLVM_ROOT
unset GLU_ROOT
unset LOCAL_LIBS

# 5. Activate the Isaac Gym compatible Python 3.8 env
source /home/sviswasam/miniforge3/etc/profile.d/conda.sh
conda activate ig_train

# 6. Add ModuMorph to Python path so metamorph, modular, graphs are importable
#    without needing pip install in ig_train env
export PYTHONPATH=/home/sviswasam/dr/ModuMorph:$PYTHONPATH

# 7. Verify the environment before launching (helps debug if job fails)
echo "Python: $(which python) $(python --version 2>&1)"
echo "CUDA: $(nvidia-smi | head -3)"
python -c "from isaacgym import gymapi, gymtorch; print('isaacgym ok')"
python -c "import metamorph; print('metamorph ok')"

# --- END ISAACGYM CONFIG ---

python -u tools/train_ppo.py --cfg ./configs/ft_g1.yaml \
    OUT_DIR ./output_walk_isaac \
    ENV.WALKER_DIR ./modular/unitree_g1_same \
    RNG_SEED 1409 \
    LOG_PERIOD 10 \
    PPO.MAX_ITERS 3000 \
    PPO.EARLY_EXIT_MAX_ITERS 3000 \
    VECENV.TYPE IsaacGym \
    PPO.NUM_ENVS 4096 \
    MODEL.FINETUNE.FULL_MODEL True \
    MODEL.TRANSFORMER.POS_EMBEDDING None \
    MODEL.TRANSFORMER.EMBEDDING_DROPOUT False \
    MODEL.TRANSFORMER.FIX_ATTENTION True \
    MODEL.TRANSFORMER.HYPERNET False \
    MODEL.TRANSFORMER.CONTEXT_ENCODER linear \
    MODEL.GRAPH_ENCODING topological | tee unitree_isaac.log