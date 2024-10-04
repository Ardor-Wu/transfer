#!/bin/bash
# run on gpu 2-3

# Function to clean up background jobs
cleanup() {
    echo "Interrupt received. Killing background processes..."
    kill -- -$$
    exit 1
}

# Set trap to catch SIGINT
trap 'cleanup' SIGINT

# Arrays of parameters
model_types=("cnn" "resnet")
training_sets=("DB" "midjourney")
ms=(20 30 64)
gpus=(2 3)  # Updated to use GPU 2 and GPU 3

# Arrays to hold commands for each GPU
declare -a cmds_gpu2
declare -a cmds_gpu3

# Job counter to distribute tasks across GPUs
job_counter=0

# Loop over parameters to generate commands
for model_type in "${model_types[@]}"; do
    for training_set in "${training_sets[@]}"; do
        for m in "${ms[@]}"; do
            # Determine GPU index
            gpu_index=$((job_counter % 2))  # Updated to use 2 GPUs (2 and 3)
            device=${gpus[$gpu_index]}
            # Generate the name parameter
            name="${model_type}_${m}_${training_set}"
            # Create the logs directory path
            log_dir="logs/$name"
            # Command to create log directory and run the python script, redirecting all output
            cmd="mkdir -p \"$log_dir\" && python train_hidden.py new --epochs 400 --batch-size 12 --name \"$name\" --model_type \"$model_type\" --message \"$m\" --dataset \"$training_set\" --gpu \"$device\" > \"$log_dir/output.log\" 2>&1"

            # Append command to the corresponding GPU array
            if [ $gpu_index -eq 0 ]; then
                cmds_gpu2+=("$cmd")  # GPU 2
            elif [ $gpu_index -eq 1 ]; then
                cmds_gpu3+=("$cmd")  # GPU 3
            fi

            # Increment job counter
            job_counter=$((job_counter + 1))
        done
    done
done

# Function to execute commands sequentially for each GPU
run_commands() {
    for cmd in "$@"; do
        echo "Running on GPU ${device}: $cmd"
        eval "$cmd"
    done
}

# Export the function for use in subshells
export -f run_commands

# Run commands for each GPU in parallel
(
    device=2
    run_commands "${cmds_gpu2[@]}"
) &

(
    device=3
    run_commands "${cmds_gpu3[@]}"
) &

# Wait for all background processes to finish
wait

echo "All tasks have been completed."
