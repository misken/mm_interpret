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
X_ldr_noq = pd.read_csv(Path(data_path, 'X_ldr_noq.csv'), index_col=0)
X_ldr_basicq = pd.read_csv(Path(data_path, 'X_ldr_basicq.csv'), index_col=0)
X_ldr_q = pd.read_csv(Path(data_path, 'X_ldr_q.csv'), index_col=0)

# y vectors
y_ldr_occ_mean = pd.read_csv(Path(data_path, 'y_ldr_occ_mean.csv'), index_col=0, squeeze=True)
y_ldr_occ_p95 = pd.read_csv(Path(data_path, 'y_ldr_occ_p95.csv'), index_col=0, squeeze=True)
y_mean_pct_blocked_by_pp = pd.read_csv(Path(data_path, 'y_mean_pct_blocked_by_pp.csv'), index_col=0, squeeze=True)
y_condmeantime_blockedbypp = pd.read_csv(Path(data_path, 'y_condmeantime_blockedbypp.csv'), index_col=0, squeeze=True)
