#!/bin/bash

# Arrays of parameters
nums=(1 2 5 10 20 30 40 50)
targets=("hidden")
pas=("mean" "median")
lengths=(20 30 64)
types=("cnn")
normalized_options=("" "--normalized")

# Set device to GPU 1
device=1

# Function to handle Ctrl+C and kill all child processes
cleanup() {
    echo "Terminating all running processes..."
    pkill -P $$
    exit 1
}

# Trap Ctrl+C (SIGINT)
trap cleanup SIGINT

# Loop over parameters to generate commands
for target in "${targets[@]}"; do
    for pa in "${pas[@]}"; do
        for n in "${nums[@]}"; do
            for length in "${lengths[@]}"; do
                for type in "${types[@]}"; do
                    for norm in "${normalized_options[@]}"; do
                        cmd="python transfer_attack.py --num_models $n --target $target --PA $pa --device $device --no_optimization $norm --target_length $length --model_type $type"
                        echo "Running on GPU ${device}: $cmd"
                        eval "$cmd"
                    done
                done
            done
        done
    done
done

echo "All tasks have been completed."
