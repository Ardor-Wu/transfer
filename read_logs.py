import os
import re
import pandas as pd


def main():
    EXP = int(input('Enter the experiment number (1, 2, or 3): '))

    # Get experiment settings
    targets, plot_directory, experiments, ks, model_types, bits_list_per_target = get_experiment_settings(EXP)

    # Base directory for the log files
    base_dir = '/scratch/qilong3/transferattack/results'

    # Read and parse logs
    df = read_and_parse_logs(base_dir, targets, experiments, bits_list_per_target, model_types, ks, EXP)

    # Process dataframe
    df = process_dataframe(df)

    # Save the DataFrame to a CSV file for later use
    df.to_csv(f'processed_data_exp{EXP}.csv', index=False)
    print(f"Processed data has been saved to 'processed_data_exp{EXP}.csv'.")


def get_experiment_settings(EXP):
    # Define the targets, ks, and experiments based on EXP
    if EXP == 1:
        targets = ['hidden', 'mbrs', 'stega', 'RivaGAN']
        plot_directory = 'plots_1'
        experiments = ['optimization']
        ks = [1, 2, 5, 10, 20, 30, 40, 50]
        model_types = ['cnn']
    elif EXP == 2:
        targets = ['hidden']
        plot_directory = 'plots_2'
        experiments = [
            'optimization',
            'mean',
            # 'median',
            # 'mean_budget_0.25_scale', 'median_budget_0.25_scale',
            'mean_budget_0.25_clamp',
            # 'median_budget_0.25_clamp',
        ]
        ks = [1, 2, 5, 10, 20, 30, 40, 50]
        model_types = ['cnn', 'resnet']
    elif EXP == 3:
        targets = ['hidden']
        plot_directory = 'plots_3'
        experiments = ['optimization'] + [f'mean_model_{i}' for i in range(2, 12)] + \
                      [f'mean_budget_0.25_clamp_model_{i}' for i in range(2, 12)]
        ks = [1, 2, 5, 10, 20, 30, 40, 50]
        model_types = ['cnn', 'resnet']
    else:
        raise ValueError('EXP must be 1, 2, or 3.')

    # Initialize bits_list_per_target
    bits_list_per_target = {}
    for target in targets:
        # Define bits_list for each target
        if target == 'hidden':
            if EXP == 1:
                bits_list = [30]
            else:
                bits_list = [20, 30, 64]
        elif target == 'mbrs':
            bits_list = [64]
        elif target == 'stega':
            bits_list = [100]
        elif target == 'RivaGAN':
            bits_list = [32]
        else:
            raise ValueError(f'Unknown target: {target}')
        bits_list_per_target[target] = bits_list

    return targets, plot_directory, experiments, ks, model_types, bits_list_per_target


def read_and_parse_logs(base_dir, targets, experiments, bits_list_per_target, model_types, ks, EXP):
    # Initialize an empty list to collect the data
    data = []
    source_bits = 30  # Assuming source bits is 30

    # Define the list of sources based on EXP
    if EXP == 1:
        sources_list = ['stable_diffusion']  # Skip 'midjourney' for EXP == 1
    else:
        sources_list = ['stable_diffusion', 'midjourney']

    # Loop over targets, bits_list, model_types, experiments, ks, and sources
    for target in targets:
        bits_list = bits_list_per_target[target]
        for bits in bits_list:
            for model_type in model_types:
                for experiment in experiments:
                    # For EXP == 3, only process k=1 for experiments other than 'optimization'
                    if EXP == 3 and experiment != 'optimization':
                        ks_to_process = [1]
                    else:
                        ks_to_process = ks
                    for k in ks_to_process:
                        for source in sources_list:
                            # Adjust base_dir depending on experiment
                            # if experiment == 'optimization':
                            #    base_dir_for_experiment = '/scratch/qilong3/transferattack/results/0.25'
                            # else:
                            #    base_dir_for_experiment = base_dir
                            # if EXP == 3 and experiment != 'optimization':
                            #    base_dir_for_experiment = base_dir
                            # else:
                            base_dir_for_experiment = '/scratch/qilong3/transferattack/results/0.25'

                            # Construct filename based on experiment and source
                            base_filename = f'hidden_to_{target}_{model_type}_{source_bits}_to_{bits}bitsAT_{k}models'
                            if source == 'midjourney':
                                base_filename += '_midjourney'
                            if experiment == 'optimization':
                                filename = base_filename + '.log'
                            else:
                                filename = base_filename + f'_no_optimization_{experiment}.log'
                            filepath = os.path.join(base_dir_for_experiment, filename)

                            if os.path.exists(filepath):
                                # Read and parse the log file
                                with open(filepath, 'r') as f:
                                    content = f.readlines()
                                    metrics = {
                                        'target': target,
                                        'k': k,
                                        'experiment': experiment,
                                        'bits': bits,
                                        'model_type': model_type,
                                        'source': 'Midjourney' if source == 'midjourney' else 'Stable Diffusion'
                                    }
                                    for line in content:
                                        # Use regex to extract metric name and value
                                        match = re.match(r'INFO:root:(.+?)\s+([0-9\.]+)', line)
                                        if match:
                                            metric_name = match.group(1).strip()
                                            value = float(match.group(2))
                                            # Sanitize the metric name to be used as a DataFrame column
                                            metric_name = metric_name.replace(' ', '_').replace('-', '_').replace(
                                                '(', '').replace(')', '')
                                            metrics[metric_name] = value
                                    data.append(metrics)
                            else:
                                print(f'File {filepath} does not exist.')

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Ensure that the 'k' column is numeric and sort the DataFrame
    df['k'] = pd.to_numeric(df['k'])
    df = df.sort_values(by=['target', 'experiment', 'bits', 'model_type', 'k', 'source'])

    return df


def process_dataframe(df):
    # Compute Evasion Rate and Evasion Rate (Unattacked)
    df['Evasion_Rate'] = 1 - df['tdr_attk']
    df['Evasion_Rate_Unattacked'] = 1 - df['tdr']

    # Return the updated DataFrame
    return df


if __name__ == '__main__':
    main()
