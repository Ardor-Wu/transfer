#!/bin/bash
# run on gpu 3

# Function to clean up background jobs
cleanup() {
    echo "Interrupt received. Killing background processes..."
    kill -- -$$
    exit 1
}

# Set trap to catch SIGINT
trap 'cleanup' SIGINT

# Arrays of parameters (easily modifiable)
model_types=("cnn")  # Modify to include other model types if needed
# model_types=("cnn" "resnet")
training_sets=("midjourney")  # Modify to include other training sets if needed
# training_sets=("midjourney" "DB")
gpus=(3)  # Modify to use different GPUs if needed

# Initialize command arrays for each GPU
for gpu in "${gpus[@]}"; do
    declare -a "cmds_gpu$gpu=()"
done

# Job counter to distribute tasks across GPUs
job_counter=0

# Loop over parameters to generate commands
for model_type in "${model_types[@]}"; do

    # Set ms depending on the model_type
    if [ "$model_type" == "cnn" ]; then
        ms=(64)  # Only 64 bits for CNN
    else
        # For other model types, specify ms values as needed
        ms=(20 30 64)
    fi

    for training_set in "${training_sets[@]}"; do
        for m in "${ms[@]}"; do

            # Determine GPU index
            gpu_index=$((job_counter % ${#gpus[@]}))
            device=${gpus[$gpu_index]}

            # Initialize extra parameters and name suffix
            extra_params=""
            name_suffix=""

            # Add loss parameters and adjust name only for 64-bit CNN
            if [ "$model_type" == "cnn" ] && [ "$m" == "64" ]; then
                extra_params="--decoder_loss 8 --encoder_loss 0.175"
                name_suffix="_dl8_el0.175"
            fi

            # Generate the name parameter
            name="${model_type}_${m}_${training_set}${name_suffix}"

            # Create the logs directory path
            log_dir="logs/$name"

            # Command to create log directory and run the python script, redirecting all output
            cmd="mkdir -p \"$log_dir\" && python train_hidden.py new --epochs 400 --batch-size 12 --name \"$name\" --model_type \"$model_type\" --message \"$m\" --dataset \"$training_set\" --gpu \"$device\" $extra_params > \"$log_dir/output.log\" 2>&1"

            # Append command to the corresponding GPU array
            declare -n cmds_array="cmds_gpu${device}"
            cmds_array+=( "$cmd" )

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
for gpu in "${gpus[@]}"; do
    cmds_var="cmds_gpu${gpu}[@]"
    (
        device=$gpu
        run_commands "${!cmds_var}"
    ) &
done

# Wait for all background processes to finish
wait

echo "All tasks have been completed."
