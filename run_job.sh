#!/bin/bash
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --time=02:00:00          # Time limit
#SBATCH --mem=16G                # Memory
#SBATCH --partition=gpu-v100     # GPU partition
#SBATCH --gres=gpu:1             # Request 1 GPU

# Load necessary modules
module load python/3.12.5
module load cuda/12.1.1

# Activate the Conda environment
source /home/kirsten.andresen/miniforge3/etc/profile.d/conda.sh
conda activate /work/forkert_lab/kirsten_andresen/conda_folder/CTA_env

# Define paths for logs and outputs
output_dir="outputs/job_${SLURM_JOB_ID}"
mkdir -p $output_dir

# Paths for input and output
input_path="/work/forkert_lab/isles24_data/preprocessed/0001"
output_path="$output_dir"

# Run the Python inference script
python /work/forkert_lab/kirsten_andresen/ISLES2024-MIPLAB-CTA/inference.py \
    --input_path $input_path \
    --output_path $output_path > $output_dir/output.log 2>&1

# Log the status of the Python script
if [ $? -eq 0 ]; then
    echo "Inference completed successfully." > $output_dir/status.log
else
    echo "Inference failed. Check output.log for details." > $output_dir/status.log
fi

# Deactivate the Conda environment
conda deactivate
