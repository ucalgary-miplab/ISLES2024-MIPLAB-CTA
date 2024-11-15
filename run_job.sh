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

source /home/kirsten.andresen/miniforge3/etc/profile.d/conda.sh
conda activate /work/forkert_lab/kirsten_andresen/conda_folder/CTA_env


python -c "import sys; print(sys.executable)"
python -c "import SimpleITK; print('SimpleITK is ready')"


# Define the output directory for logs and results
output_dir="outputs/job_${SLURM_JOB_ID}"
mkdir -p $output_dir

# Define paths
input_path="/work/forkert_lab/isles24_data/preprocessed/0001"
output_path="$output_dir"

# Run the Python inference script and save logs
python /work/forkert_lab/kirsten_andresen/ISLES2024-MIPLAB-CTA/inference.py \
    --input_path $input_path \
    --output_path $output_path > $output_dir/output.log

# Check if the Python script ran successfully and log the status
if [ $? -eq 0 ]; then
    echo "Job completed successfully." > $output_dir/status.log
else
    echo "Job failed." > $output_dir/status.log
fi

# Deactivate the Conda environment
conda deactivate
