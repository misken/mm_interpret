from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from mmfitting import crossval_summarize_mm

plt.ioff()

experiment = "exp11"
data_path = Path("data")
output_path = Path("output")
figures_path = Path("output", "figures")
raw_data_path = Path("data", "raw")
pickle_filename = f"qng_results_{experiment}.pkl"

# X matrices
X_ldr_q = pd.read_csv(Path(data_path, f'X_ldr_q_{experiment}.csv'), index_col=0)

# y vectors
y_ldr_occ_mean = pd.read_csv(Path(data_path, f'y_ldr_occ_mean_{experiment}.csv'), index_col=0, squeeze=True)
y_ldr_occ_p95 = pd.read_csv(Path(data_path, f'y_ldr_occ_p95_{experiment}.csv'), index_col=0, squeeze=True)
y_mean_pct_blocked_by_pp = pd.read_csv(Path(data_path, f'y_mean_pct_blocked_by_pp_{experiment}.csv'), index_col=0, squeeze=True)
y_condmeantime_blockedbypp = pd.read_csv(Path(data_path, f'y_condmeantime_blockedbypp_{experiment}.csv'), index_col=0, squeeze=True)

# col_idx_arate=None, col_idx_meansvctime=None, col_idx_numservers=None, load_pctile=0.95

ldr_occ_mean_q_load_results = \
    crossval_summarize_mm('ldr_occ_mean_q_load', 'ldr', 'occ_mean', X_ldr_q, y_ldr_occ_mean, scale=False,
                          flavor='load', col_idx_arate=0, col_idx_meansvctime=2)

ldr_occ_p95_q_sqrtload_results = \
    crossval_summarize_mm('ldr_occ_p95_q_sqrtload', 'ldr', 'occ_mean', X_ldr_q, y_ldr_occ_p95, scale=False,
                          flavor='sqrtload', col_idx_arate=0, col_idx_meansvctime=2, load_pctile=0.95)

mean_pct_blocked_by_pp_q_erlangc_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_q_erlangc', 'ldr', 'pct_blocked_by_pp',
                          X_ldr_q, y_mean_pct_blocked_by_pp,
                          scale=False, fit_intercept=True,
                          flavor='erlangc', col_idx_arate=0, col_idx_meansvctime=2, col_idx_numservers=6)

ldr_qng_results = {'ldr_occ_mean_q_load_results': ldr_occ_mean_q_load_results,
               'ldr_occ_p95_q_sqrtload_results': ldr_occ_p95_q_sqrtload_results,
                'mean_pct_blocked_by_pp_q_erlangc_results': mean_pct_blocked_by_pp_q_erlangc_results
               }


# Pickle the results
with open(Path(output_path, pickle_filename), 'wb') as pickle_file:
    pickle.dump(ldr_qng_results, pickle_file)