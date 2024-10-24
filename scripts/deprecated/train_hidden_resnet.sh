#!/bin/bash
# Run on GPUs 0-1 in parallel

# Function to clean up background jobs
cleanup() {
    echo "Interrupt received. Killing background processes..."
    kill -- -$$
    exit 1
}

# Set trap to catch SIGINT
trap 'cleanup' SIGINT

# Arrays of parameters
model_types=("resnet")
training_sets=("DB")
ms=(20 30 64)

# Array to hold commands
declare -a cmds

# Loop over parameters to generate commands
for model_type in "${model_types[@]}"; do
    for training_set in "${training_sets[@]}"; do
        for m in "${ms[@]}"; do
            # Generate the name parameter
            name="${model_type}_${m}_${training_set}"
            # Create the logs directory path
            log_dir="logs/$name"
            # Determine which GPU to use
            if [ "$m" -eq 64 ]; then
                gpu=1
            else
                gpu=0
            fi
            cmds+=("mkdir -p \"$log_dir\" && python train_hidden.py new --epochs 400 --batch-size 12 --name \"$name\" --model_type \"$model_type\" --message \"$m\" --dataset \"$training_set\" --gpu \"$gpu\" > \"$log_dir/output.log\" 2>&1")
        done
    done
done

# Function to execute commands in parallel
run_commands_parallel() {
    for cmd in "$@"; do
        echo "Running: $cmd"
        eval "$cmd" &
    done
    wait
}

# Run all commands in parallel
run_commands_parallel "${cmds[@]}"

echo "All tasks have been completed."
