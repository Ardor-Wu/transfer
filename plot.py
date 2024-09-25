import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Define the targets and ks
targets = ['hidden', 'mbrs']
ks = [1, 5, 10, 20, 30, 40, 60, 80, 100]

# Initialize an empty list to collect the data
data = []

# Base directory for the log files
base_dir = '/scratch/qilong3/transferattack/results'

# Loop over targets and ks to read and parse the log files
for target in targets:
    if target == 'hidden':
        bits = 30
    elif target == 'mbrs':
        bits = 64
    else:
        raise ValueError(f'Unknown target: {target}')
    for k in ks:
        filename = f'hidden_to_{target}_cnn_30_to_{bits}bitsAT_{k}models.log'
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.readlines()
                metrics = {'target': target, 'k': k}
                for line in content:
                    # Use regex to extract metric name and value
                    match = re.match(r'INFO:root:(.+?)\s+([0-9\.]+)', line)
                    if match:
                        metric_name = match.group(1).strip()
                        value = float(match.group(2))
                        # Sanitize the metric name to be used as a DataFrame column
                        metric_name = metric_name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                        metrics[metric_name] = value
                data.append(metrics)
        else:
            print(f'File {filepath} does not exist.')

# Create a DataFrame from the collected data
df = pd.DataFrame(data)

# Ensure that the 'k' column is numeric and sort the DataFrame
df['k'] = pd.to_numeric(df['k'])
df = df.sort_values(by=['target', 'k'])

# List of metric columns to plot
metric_columns = df.columns.difference(['target', 'k'])

# Create plots for each metric
for metric in metric_columns:
    plt.figure()
    for target in targets:
        subset = df[df['target'] == target]
        plt.plot(subset['k'], subset[metric], marker='o', label=target)
    plt.xlabel('Number of Models (k)')
    plt.ylabel(metric.replace('_', ' '))
    plt.title(f'{metric.replace("_", " ")} vs Number of Models (k)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/{metric}_vs_k.png')
    plt.close()

print("Plots have been saved to the current directory.")
