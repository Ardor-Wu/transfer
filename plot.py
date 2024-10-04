import os
import re
import pandas as pd
import matplotlib.pyplot as plt

EXP = 2
# Define the targets, ks, and experiments
if EXP == 1:
    targets = ['hidden', 'mbrs', 'stega', 'RivaGAN']
    plot_directory = 'plots_1'
    experiments = ['optimization', ]
elif EXP == 2:
    targets = ['hidden', ]
    plot_directory = 'plots_2'
    experiments = ['optimization', 'mean', 'median', 'mean_budget_0.25', 'median_budget_0.25']
ks = [1, 2, 5, 10, 20, 30, 40, 50]

# Initialize an empty list to collect the data
data = []

# Base directory for the log files
base_dir = '/scratch/qilong3/transferattack/results'

# Loop over targets, ks, and experiments to read and parse the log files
for target in targets:
    if target == 'hidden':
        bits_list = [20, 30, 64]
        source_bits = 30  # Assuming source bits is 30
        model_types = ['cnn', 'resnet']
    elif target == 'mbrs':
        bits_list = [64]
        source_bits = 30
        model_types = ['cnn']
    elif target == 'stega':
        bits_list = [100]
        source_bits = 30
        model_types = ['cnn']
    elif target == 'RivaGAN':
        bits_list = [32]
        source_bits = 30
        model_types = ['cnn']
    else:
        raise ValueError(f'Unknown target: {target}')
    for bits in bits_list:
        for model_type in model_types:
            for k in ks:
                for experiment in experiments:
                    if experiment == 'optimization':
                        filename = f'hidden_to_{target}_{model_type}_{source_bits}_to_{bits}bitsAT_{k}models.log'
                    else:
                        filename = f'hidden_to_{target}_{model_type}_{source_bits}_to_{bits}bitsAT_{k}models_no_optimization_{experiment}.log'
                    filepath = os.path.join(base_dir, filename)
                    if os.path.exists(filepath):
                        with open(filepath, 'r') as f:
                            content = f.readlines()
                            metrics = {
                                'target': target,
                                'k': k,
                                'experiment': experiment,
                                'bits': bits,
                                'model_type': model_type
                            }
                            for line in content:
                                # Use regex to extract metric name and value
                                match = re.match(r'INFO:root:(.+?)\s+([0-9\.]+)', line)
                                if match:
                                    metric_name = match.group(1).strip()
                                    value = float(match.group(2))
                                    # Sanitize the metric name to be used as a DataFrame column
                                    metric_name = metric_name.replace(' ', '_').replace('-', '_').replace('(',
                                                                                                          '').replace(
                                        ')', '')
                                    metrics[metric_name] = value
                            data.append(metrics)
                    else:
                        print(f'File {filepath} does not exist.')

# Create a DataFrame from the collected data
df = pd.DataFrame(data)

# Ensure that the 'k' column is numeric and sort the DataFrame
df['k'] = pd.to_numeric(df['k'])
df = df.sort_values(by=['target', 'experiment', 'bits', 'model_type', 'k'])

# List of metric columns to plot
metric_columns = df.columns.difference(['target', 'k', 'experiment', 'bits', 'model_type'])

# Create 'plots' directory if it doesn't exist
os.makedirs(plot_directory, exist_ok=True)

# Get unique bits and model_types
bits_list = sorted(df['bits'].unique())
model_types = sorted(df['model_type'].unique())

# Create plots for each metric
for metric in metric_columns:
    for bits in bits_list:
        for model_type in model_types:
            plt.figure()
            data_plotted = False
            ylabel = ''
            title = ''
            filename = ''
            for target in targets:
                for experiment in experiments:
                    subset = df[
                        (df['target'] == target) &
                        (df['experiment'] == experiment) &
                        (df['bits'] == bits) &
                        (df['model_type'] == model_type)
                        ]
                    if not subset.empty:
                        data_plotted = True
                        if metric == 'tdr':
                            y_values = 1 - subset[metric]
                            ylabel = 'Evasion Rate (Unattacked)'
                            title = f'Evasion Rate (Unattacked) vs Number of Models (k)\n(bits={bits}, model={model_type})'
                            filename = f'Evasion_Rate_Unattacked_vs_k_bits{bits}_model{model_type}.png'
                        elif metric == 'tdr_attk':
                            y_values = 1 - subset[metric]
                            ylabel = 'Evasion Rate'
                            title = f'Evasion Rate vs Number of Models (k)\n(bits={bits}, model={model_type})'
                            filename = f'Evasion_Rate_vs_k_bits{bits}_model{model_type}.png'
                        else:
                            y_values = subset[metric]
                            ylabel = metric.replace('_', ' ')
                            title = f'{metric.replace("_", " ")} vs Number of Models (k)\n(bits={bits}, model={model_type})'
                            filename = f'{metric}_vs_k_bits{bits}_model{model_type}.png'
                        plt.plot(subset['k'], y_values, marker='o', label=f'{target}_{experiment}')
            if data_plotted:
                plt.xlabel('Number of Models (k)')
                plt.ylabel(ylabel)
                plt.title(title)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_directory, filename))
                plt.close()
            else:
                plt.close()
                print(f'No data available for metric {metric}, bits {bits}, model_type {model_type}. Skipping plot.')

# Add new plot: Evasion Rate (Attacked) vs Average L_inf Perturbation
for bits in bits_list:
    for model_type in model_types:
        plt.figure()
        data_plotted = False
        for target in targets:
            for experiment in experiments:
                subset = df[
                    (df['target'] == target) &
                    (df['experiment'] == experiment) &
                    (df['bits'] == bits) &
                    (df['model_type'] == model_type)
                    ]
                if not subset.empty:
                    data_plotted = True
                    # Calculate Evasion Rate (Attacked) and get Average L_inf Perturbation
                    x_values = 1 - subset['tdr_attk']
                    y_values = subset['noise_L_infinity']
                    plt.plot(x_values, y_values, marker='o', label=f'{target}_{experiment}')
        if data_plotted:
            plt.xlabel('Evasion Rate')
            plt.ylabel('Average $L_\\infty$ Perturbation')
            plt.title(f'Average $L_\\infty$ Perturbation vs Evasion Rate\n(bits={bits}, model={model_type})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_directory, f'Avg_Linf_vs_Evasion_Rate_Attacked_bits{bits}_model{model_type}.png'))
            plt.close()
        else:
            plt.close()
            print(f'No data available for bits {bits}, model_type {model_type}. Skipping plot.')

print(f"Plots have been saved to the {plot_directory} directory.")
