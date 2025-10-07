import os
import json

# Replace this with the path to your directory
directory_path = './results'

# Get all JSON filenames and sort them alphabetically
json_files = sorted(f for f in os.listdir(directory_path) if f.endswith('.json'))

for filename in json_files:
    file_path = os.path.join(directory_path, filename)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            val_accs = data.get('val_accs', [])
            if isinstance(val_accs, list) and val_accs:
                max_val = max(val_accs)
                print(f"{filename}: max val_acc = {max_val}")
            else:
                print(f"{filename}: 'val_accs' is missing or empty")
    except Exception as e:
        print(f"{filename}: Failed to read or parse JSON ({e})")
