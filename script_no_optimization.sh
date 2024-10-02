
#!/bin/bash

# Arrays of parameters
nums=(1 2 5 10 20 30 40 50)
#targets=("mbrs" "hidden")
targets=("stega")
pas=("mean" "median")
gpus=(0 1 2)

# Arrays to hold commands for each GPU
declare -a cmds_gpu0
declare -a cmds_gpu1
declare -a cmds_gpu2

# Job counter to distribute tasks across GPUs
job_counter=0

# Loop over targets, pas, and nums to generate commands
for target in "${targets[@]}"; do
    for pa in "${pas[@]}"; do
        for n in "${nums[@]}"; do
            # Determine GPU index
            gpu_index=$((job_counter % 3))
            device=${gpus[$gpu_index]}
            cmd="python transfer_attack.py --num_models $n --target $target --PA $pa --device $device --no_optimization --normalized"
            # Append command to the corresponding GPU array
            if [ $gpu_index -eq 0 ]; then
                cmds_gpu0+=("$cmd")
            elif [ $gpu_index -eq 1 ]; then
                cmds_gpu1+=("$cmd")
            elif [ $gpu_index -eq 2 ]; then
                cmds_gpu2+=("$cmd")
            fi

            # Increment job counter
            job_counter=$((job_counter + 1))
        done
    done
done

# Function to execute commands sequentially for each GPU
run_commands() {
    local cmds=("$@")
    for cmd in "${cmds[@]}"; do
        echo "Running on GPU ${device}: $cmd"
        eval "$cmd"
    done
}

# Export the function and device variable for use in subshells
export -f run_commands

# Run commands for each GPU in parallel
(
    device=0
    run_commands "${cmds_gpu0[@]}"
) &

(
    device=1
    run_commands "${cmds_gpu1[@]}"
) &

(
    device=2
    run_commands "${cmds_gpu2[@]}"
) &

# Wait for all background processes to finish
wait

echo "All tasks have been completed."
