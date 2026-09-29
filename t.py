
import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/rotax_combined_clean.csv', usecols=['run_id'])
r_ids = df['run_id'].unique()

print(f'min run_id: {r_ids.min()}')
print(f'max run_id: {r_ids.max()}')
print(f'unique run_ids: {len(r_ids)}')

bad = [x for x in r_ids if not (isinstance(x, (int, float, np.integer, np.floating)) and x >= 0 and x == int(x))]
if not bad:
    print('All run_ids are non-negative integers: True')
else:
    print(f'Found incompatible run_ids: {bad}')

import os, re
pattern = re.compile(r'0x1[0-9a-fA-F]{2}')
for root, dirs, files in os.walk('.'):
    if 'node_modules' in root or '.git' in root or '__pycache__' in root:
        continue
    for file in files:
        if file.endswith('.py') or file.endswith('.md') or file.endswith('.txt'):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                matches = pattern.findall(content)
                if matches:
                    print(f'{path}: {set(matches)}')
