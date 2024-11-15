#!/bin/bash
#SBATCH --nodes=1               
#SBATCH --ntasks=1               
#SBATCH --time=02:00:00          
#SBATCH --mem=16G                
#SBATCH --partition=gpu-v100     
#SBATCH --gres=gpu:1             

# Load necessary modules
module load python/3.12.5
module load cuda/12.1.1

# Activate the Conda environment
source /home/kirsten.andresen/miniforge3/etc/profile.d/conda.sh
conda activate /work/forkert_lab/kirsten_andresen/conda_folder/CTA_env

# Navigate to your working directory
cd /work/forkert_lab/kirsten_andresen

# Define the output directory for logs and results
output_dir="inference_outputs/job_${SLURM_JOB_ID}"
mkdir -p $output_dir

# Check PyTorch and CUDA setup, log the output
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')" >> $output_dir/setup_check.log

# Define paths
input_path="/work/forkert_lab/isles24_data/preprocessed/patient_001"
output_path="$output_dir"

# Run the Python inference script and save logs
python /work/forkert_lab/kirsten_andresen/scripts/inference.py \
    --input_path $input_path \
    --output_path $output_path > $output_dir/output.log

# Check if the Python script ran successfully and log the status
if [ $? -eq 0 ]; then
    echo "Inference completed successfully." > $output_dir/status.log
else
    echo "Inference failed." > $output_dir/status.log
fi

# Deactivate the Conda environment
conda deactivate
