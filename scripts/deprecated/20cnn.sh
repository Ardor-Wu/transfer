#!/bin/bash

# Activate the virtual environment
source /scratch/qilong3/anaconda3/bin/activate transfer

# Arrays of parameters
nums=(1 2 5 10 20 30 40 50)
targets=("hidden")
lengths=(20)
types=("cnn")

# Function to handle Ctrl+C and kill all child processes
cleanup() {
    echo "Interrupt received. Terminating all running processes..."
    kill -- -$$
    exit 1
}

# Trap Ctrl+C (SIGINT)
trap cleanup SIGINT

# Array to hold commands
declare -a cmds

# Loop over parameters to generate commands
for target in "${targets[@]}"; do
    for n in "${nums[@]}"; do
        for length in "${lengths[@]}"; do
            for type in "${types[@]}"; do
                cmd="python /scratch/qilong3/transferattack/transfer_attack.py --num_models $n --target_length $length --model_type $type"
                cmds+=("$cmd")
            done
        done
    done
done

# Function to execute commands in parallel but sequential on each GPU
run_commands_parallel_sequential() {
    gpu_ids=(1 2 3)
    declare -A gpu_queues

    # Initialize queues for each GPU
    for gpu in "${gpu_ids[@]}"; do
        gpu_queues[$gpu]=""
    done

    # Distribute commands to GPU queues
    gpu_index=0
    for cmd in "$@"; do
        gpu=${gpu_ids[$gpu_index]}
        gpu_queues[$gpu]="${gpu_queues[$gpu]}$cmd;"
        gpu_index=$(( (gpu_index + 1) % ${#gpu_ids[@]} ))
    done

    # Run commands sequentially on each GPU in parallel
    for gpu in "${gpu_ids[@]}"; do
        (
            IFS=';' read -r -a commands <<< "${gpu_queues[$gpu]}"
            for cmd in "${commands[@]}"; do
                if [ -n "$cmd" ]; then
                    echo "Running on GPU $gpu: $cmd"
                    CUDA_VISIBLE_DEVICES="$gpu" eval "$cmd"
                fi
            done
        ) &
    done

    wait
}

# Run all commands
run_commands_parallel_sequential "${cmds[@]}"

echo "All tasks have been completed."