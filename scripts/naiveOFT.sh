#!/bin/bash

# Function to clean up background jobs
cleanup() {
    echo "Interrupt received. Killing background processes..."
    kill -- -$$
    exit 1
}

# Set trap to catch SIGINT
trap 'cleanup' SIGINT

# Arrays of parameters
nums=(1)
#targets=("RivaGAN" "mbrs" "stega")
targets=("mbrs")
gpus=(0 1 2 3)  # GPUs to use
max_parallel_jobs=${#gpus[@]}  # Set max parallel jobs to the number of GPUs
normalized_values=("--normalized" "")  # Values for --normalized

# Set target lengths for each target
declare -A target_lengths=(
    ["RivaGAN"]="32"
    ["mbrs"]="64 256"
    ["hidden"]="30"
    ["stega"]="100"
)

# Function to wait for available job slots
wait_for_jobs() {
    while (( $(jobs -p | wc -l) >= max_parallel_jobs )); do
        sleep 1
    done
}

# Keep track of the task index for GPU assignment
task_index=0

# Loop over targets, nums, and --normalized to create tasks
for target in "${targets[@]}"; do
    lengths="${target_lengths[$target]}"
    # Loop over target lengths
    for target_length in $lengths; do
        for n in "${nums[@]}"; do
            for normalized in "${normalized_values[@]}"; do
                wait_for_jobs  # Ensure we don't exceed max_parallel_jobs
                gpu_id=${gpus[$((task_index % ${#gpus[@]}))]}  # Assign GPU in round-robin
                cmd="CUDA_VISIBLE_DEVICES=$gpu_id python transfer_attack.py --num_models $n --target $target --target_length $target_length $normalized --no_optimization"
                echo "Running on GPU $gpu_id: $cmd"
                eval "$cmd" &
                task_index=$((task_index + 1))
            done
        done
    done
done

# Wait for all background processes to finish
wait

echo "All tasks have been completed."
