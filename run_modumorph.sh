#!/usr/bin/env bash
#SBATCH --mail-user=sviswasam@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -t 00:10:00
#SBATCH --mem 16G
#SBATCH --job-name="train"

#SBATCH --output=/home/sviswasam/dr/ModuMorph/logs/output_stand.6log
#SBATCH --error=/home/sviswasam/dr/ModuMorph/logs/err_stand6.err

# --- START MUJOCO CONFIG ---
# 0. Initialize Module System
source /etc/profile.d/modules.sh

# 1. Load Modules
module load cuda/11.1.0/pdhgaya
module load libx11/1.8.12/wtcqjwl
module load glew/2.2.0/azi6l2x

# 2. DEFINE PATHS (Using Absolute Paths for Safety)
export GLU_ROOT=/cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/mesa-glu-9.0.2-fzzvroqpxdyho2fdiav4wk23ueazizkh
export LLVM_ROOT=/cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/llvm-18.1.8-bpqadvig7ku5rfx4jckqwuhf6lk5uljq
export LOCAL_LIBS=/home/sviswasam/dr/local_libs

# 3. FORCE LOAD OPENGL (The Nuclear Fix)
# This forces the system to grab your local file immediately.
export LD_PRELOAD=$LOCAL_LIBS/libOpenGL.so.0

# 4. RUNTIME LIBRARY PATHS
export LD_LIBRARY_PATH=$LOCAL_LIBS:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LLVM_ROOT/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GLU_ROOT/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cm/shared/spack/opt/spack/linux-x86_64/mesa-25.0.5-rclmohiaitjskxy2qh4gqxqi2wt5hv75/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cm/shared/spack/opt/spack/linux-x86_64/glew-2.2.0-azi6l2xafyqw4k4c46rel6bbf5tjon6v/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sviswasam/.mujoco/mujoco210/bin:/usr/lib/nvidia

# 5. COMPILER FLAGS & INCLUDES
export CPATH=$CPATH:$GLU_ROOT/include
export CPATH=$CPATH:/cm/shared/spack/opt/spack/linux-x86_64/libx11-1.8.12-wtcqjwlazka4hmy6zdqc7nmzowe6omgr/include
export CPATH=$CPATH:/cm/shared/spack/opt/spack/linux-x86_64/xproto-7.0.31-lih7ldnzfw22idompow35rqkyqeo6gay/include
export CPATH=$CPATH:/cm/shared/spack/opt/spack/linux-x86_64/glew-2.2.0-azi6l2xafyqw4k4c46rel6bbf5tjon6v/include
export CPATH=$CPATH:/cm/shared/spack/opt/spack/linux-x86_64/mesa-25.0.5-rclmohiaitjskxy2qh4gqxqi2wt5hv75/include
export C_INCLUDE_PATH=$CPATH

export CFLAGS="-Wno-error"
# --- END MUJOCO CONFIG ---

# Python Path to identify the right metamorph
export PYTHONPATH=/home/sviswasam/dr/ModuMorph:$PYTHONPATH

# Activate env
source /home/sviswasam/dr/modumorph_env/bin/activate

python -u tools/train_ppo.py --cfg ./configs/ft_g1.yaml \
    OUT_DIR ./output_stand6 \
    ENV.WALKER_DIR ./modular/unitree_g1_actual \
    RNG_SEED 1409 \
    LOG_PERIOD 10 \
    PPO.MAX_ITERS 5000 \
    PPO.EARLY_EXIT_MAX_ITERS 5000 \
    MODEL.FINETUNE.FULL_MODEL True \
    MODEL.TRANSFORMER.POS_EMBEDDING None \
    MODEL.TRANSFORMER.EMBEDDING_DROPOUT False \
    MODEL.TRANSFORMER.FIX_ATTENTION True \
    MODEL.TRANSFORMER.HYPERNET True \
    MODEL.TRANSFORMER.CONTEXT_ENCODER linear \
    MODEL.GRAPH_ENCODING topological | tee unitree_top6.log