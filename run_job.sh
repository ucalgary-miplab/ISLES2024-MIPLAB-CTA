#!/bin/bash
#SBATCH --nodes=1               
#SBATCH --ntasks=1             
#SBATCH --time=02:00:00          
#SBATCH --mem=16G                
#SBATCH --partition=gpu-v100     
#SBATCH --gres=gpu:1            

# Ensure a clean environment
module purge

# Load necessary CUDA module (do not load Python module to avoid conflicts)
module load cuda/12.1.1

# Activate the Conda environment
source /home/kirsten.andresen/miniforge3/etc/profile.d/conda.sh
conda activate /work/forkert_lab/kirsten_andresen/conda_folder/CTA_env

# Verify the environment (optional debug logs)
echo "Python Executable: $(which python)"
python -c "import sys; print('Python Path:', sys.path)"
python -c "import torch; print('Torch Version:', torch.__version__)"

# Define paths for logs and outputs
output_dir="outputs/job_${SLURM_JOB_ID}"
mkdir -p $output_dir

# Run the Python inference script
python /work/forkert_lab/kirsten_andresen/ISLES2024-MIPLAB-CTA/inference.py > $output_dir/output.log 2>&1

# Log the status of the Python script
if [ $? -eq 0 ]; then
    echo "Inference completed successfully." > $output_dir/status.log
else
    echo "Inference failed. Check output.log for details." > $output_dir/status.log
fi

# Deactivate the Conda environment
conda deactivate
