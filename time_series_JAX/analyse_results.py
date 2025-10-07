import os
import numpy as np
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-task", type=str, required=True, help="Task name")
args = parser.parse_args()

# Root directory containing all result directories
root_dir = f'./outputs/pdssm/{args.task}'  # <-- Replace with your actual path

# Dictionary to store max values grouped by hyperparameter configuration
results = defaultdict(list)

didnt_find=0
not_yet=0

# Iterate through all subdirectories
for dirname in os.listdir(root_dir):
    dirpath = os.path.join(root_dir, dirname)
    if not os.path.isdir(dirpath):
        continue

    # Assume last 4 characters are the seed
    config_name = dirname[:-4]

    test_metric_path = os.path.join(dirpath, 'test_metric.npy')
    if not os.path.isfile(test_metric_path):
        #print(f"Warning: test_metric.npy not found in {dirpath}")
        didnt_find += 1
        continue

    try:
        metric_array = np.load(test_metric_path)
        max_metric = np.max(metric_array)
        results[config_name].append(max_metric)
    except Exception as e:
        print(f"Error reading {test_metric_path}: {e}")

# Report results

for config, metrics in results.items():
    if len(metrics) < 5:
        print(f"Only found {len(metrics)} runs for {config}")
        not_yet+=1
        continue

best_mean = -1
best_config = None
for config, metrics in results.items():
    if len(metrics) < 5:
        continue
    mean = np.mean(metrics)
    std = np.std(metrics)
    if mean > best_mean:
        best_config = config
        best_mean = mean
        best_std = std

print('\n')
print(f'{didnt_find} runs did not produce test_metric.npy')
print(f'{not_yet} runs do not yet have 5 seeds evaluated.')

print(f"Best config: {best_config} | Mean Accuracy: {100*best_mean:.4f} | Std Dev: {100*best_std:.4f}")