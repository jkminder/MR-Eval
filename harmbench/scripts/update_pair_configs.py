"""
One-shot script to update PAIR configs on the cluster.
Run via: curl -fsSL <raw_url> | python3
"""
import urllib.request, os

WORKSPACE = '/mnt/dlabscratch1/moskvore/MR-Eval'
REPO = 'VityaVitalich/MR-Eval'
BRANCH = 'main'

paths = [
    'harmbench/configs/method_configs/PAIR_config.yaml',
    'harmbench/configs/pipeline_configs/run_pipeline_text.yaml',
    'harmbench/plot_pair_dynamics.py',
    'harmbench/plot_pair_score_comparison.py',
]

for path in paths:
    url = f'https://raw.githubusercontent.com/{REPO}/{BRANCH}/{path}'
    dest = f'{WORKSPACE}/{path}'
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f'Fetching {path} ...', end=' ', flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        print('OK')
    except Exception as e:
        print(f'FAILED: {e}')

print('Done.')
