from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from mmfitting import crossval_summarize_mm

data_path = Path("data")
output_path = Path("output")
figures_path = Path("output", "figures")
raw_data_path = Path("data", "raw")

# X matrices
X_obs_noq = pd.read_csv(Path(data_path, 'X_obs_noq.csv'), index_col=0)
X_obs_basicq = pd.read_csv(Path(data_path, 'X_obs_basicq.csv'), index_col=0)
X_obs_q = pd.read_csv(Path(data_path, 'X_obs_q.csv'), index_col=0)

# y vectors
y_obs_occ_mean = pd.read_csv(Path(data_path, 'y_obs_occ_mean.csv'), index_col=0, squeeze=True)
y_obs_occ_p95 = pd.read_csv(Path(data_path, 'y_obs_occ_p95.csv'), index_col=0, squeeze=True)
y_mean_pct_blocked_by_ldr = pd.read_csv(Path(data_path, 'y_mean_pct_blocked_by_ldr.csv'), index_col=0, squeeze=True)
y_condmeantime_blockedbyldr = pd.read_csv(Path(data_path, 'y_condmeantime_blockedbyldr.csv'), index_col=0, squeeze=True)