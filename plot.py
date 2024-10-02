import os
import re
import pandas as pd
import matplotlib.pyplot as plt

EXP = 1
# Define the targets, ks, and experiments
if EXP == 1:
    targets = ['hidden', 'mbrs', 'stega']
    plot_directory = 'plots_1'
    experiments = ['optimization', ]
elif EXP == 2:
    targets = ['hidden', ]
    plot_directory = 'plots_2'
    experiments = ['optimization', 'mean', 'median', 'mean_budget_0.25', 'median_budget_0.25']
ks = [1, 2, 5, 10, 20, 30, 40, 50]

# experiments = ['optimization']
# Initialize an empty list to collect the data
data = []

# Base directory for the log files
base_dir = '/scratch/qilong3/transferattack/results'

# Loop over targets, ks, and experiments to read and parse the log files
for target in targets:
    if target == 'hidden':
        bits = 30
    elif target == 'mbrs':
        bits = 64
    elif target == 'stega':
        bits = 100
    else:
        raise ValueError(f'Unknown target: {target}')
    for k in ks:
        for experiment in experiments:
            if experiment == 'optimization':
                filename = f'hidden_to_{target}_cnn_30_to_{bits}bitsAT_{k}models.log'
            else:
                filename = f'hidden_to_{target}_cnn_30_to_{bits}bitsAT_{k}models_no_optimization_{experiment}.log'
            filepath = os.path.join(base_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    content = f.readlines()
                    metrics = {'target': target, 'k': k, 'experiment': experiment}
                    for line in content:
                        # Use regex to extract metric name and value
                        match = re.match(r'INFO:root:(.+?)\s+([0-9\.]+)', line)
                        if match:
                            metric_name = match.group(1).strip()
                            value = float(match.group(2))
                            # Sanitize the metric name to be used as a DataFrame column
                            metric_name = metric_name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')',
                                                                                                                   '')
                            metrics[metric_name] = value
                    data.append(metrics)
            else:
                print(f'File {filepath} does not exist.')

# Create a DataFrame from the collected data
df = pd.DataFrame(data)

# Ensure that the 'k' column is numeric and sort the DataFrame
df['k'] = pd.to_numeric(df['k'])
df = df.sort_values(by=['target', 'experiment', 'k'])

# List of metric columns to plot
metric_columns = df.columns.difference(['target', 'k', 'experiment'])

# Create 'plots' directory if it doesn't exist
os.makedirs(plot_directory, exist_ok=True)

# Create plots for each metric
for metric in metric_columns:
    plt.figure()
    for target in targets:
        for experiment in experiments:
            subset = df[(df['target'] == target) & (df['experiment'] == experiment)]
            if not subset.empty:
                if metric == 'tdr':
                    y_values = 1 - subset[metric]
                    ylabel = 'Evasion Rate (Unattacked)'
                    title = 'Evasion Rate (Unattacked) vs Number of Models (k)'
                    filename = 'Evasion_Rate_Unattacked_vs_k.png'
                elif metric == 'tdr_attk':
                    y_values = 1 - subset[metric]
                    ylabel = 'Evasion Rate'
                    title = 'Evasion Rate vs Number of Models (k)'
                    filename = 'Evasion_Rate_vs_k.png'
                else:
                    y_values = subset[metric]
                    ylabel = metric.replace('_', ' ')
                    title = f'{metric.replace("_", " ")} vs Number of Models (k)'
                    filename = f'{metric}_vs_k.png'
                plt.plot(subset['k'], y_values, marker='o', label=f'{target}_{experiment}')
    plt.xlabel('Number of Models (k)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_directory, filename))
    plt.close()

# Add new plot: Evasion Rate (Attacked) vs Average L_inf Perturbation
plt.figure()
for target in targets:
    for experiment in experiments:
        subset = df[(df['target'] == target) & (df['experiment'] == experiment)]
        if not subset.empty:
            # Calculate Evasion Rate (Attacked) and get Average L_inf Perturbation
            x_values = 1 - subset['tdr_attk']
            y_values = subset['noise_L_infinity']
            plt.plot(x_values, y_values, marker='o', label=f'{target}_{experiment}')
plt.xlabel('Evasion Rate')
plt.ylabel('Average $L_\infty$ Perturbation')
plt.title('Average $L_\infty$ Perturbation vs Evasion Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_directory, 'Avg_Linf_vs_Evasion_Rate_Attacked.png'))
plt.close()

print(f"Plots have been saved to the {plot_directory} directory.")
