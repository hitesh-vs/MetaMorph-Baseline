#!/usr/bin/env bash
# Script for creating a pkl file with trajectories so that video can be created locally

#SBATCH --mail-user=sviswasam@wpi.edu
#SBATCH --mail-type=ALL
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 00:05:00
#SBATCH --mem 16G
#SBATCH --job-name="video"

# Log Names
#SBATCH --output=/home/sviswasam/dr/metamorph/logs/video_out3.log
#SBATCH --error=/home/sviswasam/dr/metamorph/logs/video_err3.err

# --- START MUJOCO CONFIG ---
module load cuda/11.1.0/pdhgaya
module load libx11/1.8.12/wtcqjwl
module load glew/2.2.0/azi6l2x

# 1. PATH DEFINITIONS
export MESA_ROOT=/cm/shared/spack/opt/spack/linux-x86_64/mesa-25.0.5-rclmohiaitjskxy2qh4gqxqi2wt5hv75
export GLU_ROOT=/cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/mesa-glu-9.0.2-fzzvroqpxdyho2fdiav4wk23ueazizkh
export GLEW_ROOT=/cm/shared/spack/opt/spack/linux-x86_64/glew-2.2.0-azi6l2xafyqw4k4c46rel6bbf5tjon6v
# Point to YOUR local libs folder (adjust path if it's not in metamorph)
export LOCAL_LIB_DIR=/home/sviswasam/dr/local_libs

# 2. RENDERING PREFS
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_EGL_DEVICE_ID=-1

# 3. RUNTIME LIBRARY PATHS
# We put LOCAL_LIB_DIR at the very beginning so the system finds your libOpenGL.so.0 first
export LD_LIBRARY_PATH=$LOCAL_LIB_DIR:$MESA_ROOT/lib:$GLU_ROOT/lib:$GLEW_ROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sviswasam/.mujoco/mujoco210/bin

# 4. PRELOAD (Directly referencing your local file)
export LD_PRELOAD=$MESA_ROOT/lib/libOSMesa.so:$MESA_ROOT/lib/libGL.so.1:$LOCAL_LIB_DIR/libOpenGL.so.0

# Set offscreen rendering
export MUJOCO_GL=osmesa

# --- END MUJOCO CONFIG ---

source /home/sviswasam/dr/metamorph_env/bin/activate

# 5. VERIFICATION (Check if the file exists before running)
if [ ! -f "$LOCAL_LIB_DIR/libOpenGL.so.0" ]; then
    echo "ERROR: libOpenGL.so.0 not found in $LOCAL_LIB_DIR"
    exit 1
fi

# 6. EXECUTION
echo "Starting Video gen"
python -u tools/render_video.py
echo "Video generation complete!"