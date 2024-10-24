#!/bin/bash

# for extended exp 2

# Arrays of parameters
nums=(1)
targets=("hidden")
pas=("mean")  # Keep mean only
normalized_options=("" "--normalized")
normalization_methods=("clamp")  # Keep clamp only
data_names=("DB" "midjourney")  # New array for data names

# Function to handle Ctrl+C and kill all child processes
cleanup() {
    echo "Terminating all running processes..."
    pkill -P $$
    exit 1
}

# Trap Ctrl+C (SIGINT)
trap cleanup SIGINT

# List of GPUs to use
gpus=(2)

# Initialize associative array to hold commands per GPU
declare -A commands_per_gpu

for gpu in "${gpus[@]}"; do
    commands_per_gpu[$gpu]=""
done

# Initialize gpu_index
gpu_index=0

# Generate the list of commands, assigning them to GPUs
for target in "${targets[@]}"; do
    for pa in "${pas[@]}"; do
        for n in "${nums[@]}"; do
            for data_name in "${data_names[@]}"; do  # Loop over data_names
                # Set types and lengths depending on data_name
                if [ "$data_name" == "DB" ]; then
                    types=("cnn")       # For DB, only 'cnn'
                    lengths=(64)        # For DB, only length 64
                else
                    types=("cnn" "resnet")  # For midjourney, both 'cnn' and 'resnet'
                    lengths=(20 30 64)      # For midjourney, lengths 20, 30, 64
                fi
                for length in "${lengths[@]}"; do
                    for type in "${types[@]}"; do
                        for norm_flag in "${normalized_options[@]}"; do
                            # When --normalized is included, loop over normalization methods
                            for norm in "${normalization_methods[@]}"; do
                                for model_index in {2..11}; do
                                    gpu=${gpus[$gpu_index]}
                                    cmd="python transfer_attack.py --num_models $n --target $target --PA $pa --device $gpu --no_optimization"
                                    # Add norm_flag and normalization method if norm_flag is not empty
                                    if [ -n "$norm_flag" ]; then
                                        cmd+=" $norm_flag --normalization $norm"
                                    fi
                                    cmd+=" --target_length $length --model_type $type --model_index $model_index --data_name $data_name"
                                    commands_per_gpu[$gpu]+="$cmd"$'\n'
                                    # Increment gpu_index modulo number of GPUs
                                    gpu_index=$(( (gpu_index + 1) % ${#gpus[@]} ))
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# Function to run commands sequentially
run_commands() {
    local commands=("$@")
    for cmd in "${commands[@]}"; do
        echo "$cmd"
        eval "$cmd"
    done
}

# Run commands on each GPU in parallel
pids=()

for gpu in "${gpus[@]}"; do
    if [ -n "${commands_per_gpu[$gpu]}" ]; then
        # Read the commands into an array
        IFS=$'\n' read -rd '' -a commands <<< "${commands_per_gpu[$gpu]}"
        run_commands "${commands[@]}" &
        pids+=($!)
    fi
done

# Wait for all background processes to finish
for pid in "${pids[@]}"; do
    wait $pid
done

echo "All tasks have been completed."
