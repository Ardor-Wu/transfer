#!/bin/bash
# run on gpu 0-3

# Function to clean up background jobs
cleanup() {
    echo "Interrupt received. Killing background processes..."
    kill -- -$$
    exit 1
}

# Set trap to catch SIGINT
trap 'cleanup' SIGINT

# Arrays of parameters
nums=(1 2 5 10 20 30 40 50)
targets=("RivaGAN" "mbrs" "hidden" "stega")
gpus=(0 1 2 3)

# Function to execute commands sequentially for each GPU
run_commands() {
    local gpu_id=$1
    local target=$2
    local target_length=$3
    shift 3
    local nums=("$@")

    for n in "${nums[@]}"; do
        cmd="python transfer_attack.py --num_models $n --target $target --target_length $target_length --device $gpu_id"
        echo "Running on GPU $gpu_id: $cmd"
        eval "$cmd"
    done
}

# Set target lengths for each target
declare -A target_lengths

target_lengths=(
    ["RivaGAN"]=32
    ["mbrs"]=64
    ["hidden"]=30
    ["stega"]=100
)

# Run commands for each target on a separate GPU in parallel
for i in "${!targets[@]}"; do
    target="${targets[$i]}"
    target_length="${target_lengths[$target]}"
    gpu_id=${gpus[$i]}
    run_commands $gpu_id "$target" "$target_length" "${nums[@]}" &
done

# Wait for all background processes to finish
wait

echo "All tasks have been completed."