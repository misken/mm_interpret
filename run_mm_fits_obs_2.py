from pathlib import Path
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from mmfitting import crossval_summarize_mm

experiment = "exp10obflow06"
data_path = Path("data")
output_path = Path("output")
figures_path = Path("output", "figures")
raw_data_path = Path("data", "raw")

# X matrices
X_obs_noq = pd.read_csv(Path(data_path, f'X_obs_noq_{experiment}.csv'), index_col=0)
X_obs_basicq = pd.read_csv(Path(data_path, f'X_obs_basicq_{experiment}.csv'), index_col=0)
X_obs_q = pd.read_csv(Path(data_path, f'X_obs_q_{experiment}.csv'), index_col=0)

# y vectors
y_obs_occ_mean = pd.read_csv(Path(data_path, f'y_obs_occ_mean_{experiment}.csv'), index_col=0, squeeze=True)
y_obs_occ_p95 = pd.read_csv(Path(data_path, f'y_obs_occ_p95_{experiment}.csv'), index_col=0, squeeze=True)
y_mean_pct_blocked_by_ldr = pd.read_csv(Path(data_path, f'y_mean_pct_blocked_by_ldr_{experiment}.csv'), index_col=0, squeeze=True)
y_condmeantime_blockedbyldr = pd.read_csv(Path(data_path, f'y_condmeantime_blockedbyldr_{experiment}.csv'), index_col=0, squeeze=True)

## Linear regression (lm)


mean_pct_blocked_by_ldr_basicq_lm_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_q_lm', 'obs', 'pct_blocked_by_ldr',
                          X_obs_basicq, y_mean_pct_blocked_by_ldr,
                                               scale=False, fit_intercept=True, flavor='lm')

mean_pct_blocked_by_ldr_q_lm_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_basicq_lm', 'obs', 'pct_blocked_by_ldr',
                          X_obs_q, y_mean_pct_blocked_by_ldr,
                                               scale=False, fit_intercept=True, flavor='lm')

mean_pct_blocked_by_ldr_noq_lm_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_noq_lm', 'obs', 'pct_blocked_by_ldr',
                          X_obs_noq, y_mean_pct_blocked_by_ldr,
                                               scale=False, fit_intercept=True, flavor='lm')

condmeantime_blockedbyldr_basicq_lm_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_basicq_lm', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_basicq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, fit_intercept=True, flavor='lm')

condmeantime_blockedbyldr_q_lm_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_q_lm', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_q.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, fit_intercept=True, flavor='lm')

condmeantime_blockedbyldr_noq_lm_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_noq_lm', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_noq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, fit_intercept=True, flavor='lm')

# LassoCV (lassocv)


mean_pct_blocked_by_ldr_basicq_lassocv_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_q_lassocv', 'obs', 'pct_blocked_by_ldr',
                          X_obs_basicq, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)

mean_pct_blocked_by_ldr_q_lassocv_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_basicq_lassocv', 'obs', 'pct_blocked_by_ldr',
                          X_obs_q, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)

mean_pct_blocked_by_ldr_noq_lassocv_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_noq_lassocv', 'obs', 'pct_blocked_by_ldr',
                          X_obs_noq, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)


condmeantime_blockedbyldr_basicq_lassocv_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_basicq_lassocv', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_basicq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)

condmeantime_blockedbyldr_q_lassocv_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_q_lassocv', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_q.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)

condmeantime_blockedbyldr_noq_lassocv_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_noq_lassocv', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_noq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)


# Polynomial regression (poly)


mean_pct_blocked_by_ldr_basicq_poly_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_q_poly', 'obs', 'pct_blocked_by_ldr',
                          X_obs_basicq, y_mean_pct_blocked_by_ldr,
                                               scale=False, flavor='lm')

mean_pct_blocked_by_ldr_q_poly_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_basicq_poly', 'obs', 'pct_blocked_by_ldr',
                          X_obs_q, y_mean_pct_blocked_by_ldr,
                                               scale=False, flavor='lm')

mean_pct_blocked_by_ldr_noq_poly_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_noq_poly', 'obs', 'pct_blocked_by_ldr',
                          X_obs_noq, y_mean_pct_blocked_by_ldr,
                                               scale=False, flavor='lm')

condmeantime_blockedbyldr_basicq_poly_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_basicq_poly', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_basicq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, flavor='lm')

condmeantime_blockedbyldr_q_poly_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_q_poly', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_q.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, flavor='lm')

condmeantime_blockedbyldr_noq_poly_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_noq_poly', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_noq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, flavor='lm')

# Random forest (rf)


mean_pct_blocked_by_ldr_basicq_rf_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_q_rf', 'obs', 'pct_blocked_by_ldr',
                          X_obs_basicq, y_mean_pct_blocked_by_ldr,
                                               scale=False, flavor='rf')

mean_pct_blocked_by_ldr_q_rf_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_basicq_rf', 'obs', 'pct_blocked_by_ldr',
                          X_obs_q, y_mean_pct_blocked_by_ldr,
                                               scale=False, flavor='rf')

mean_pct_blocked_by_ldr_noq_rf_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_noq_rf', 'obs', 'pct_blocked_by_ldr',
                          X_obs_noq, y_mean_pct_blocked_by_ldr,
                                               scale=False, flavor='rf')

condmeantime_blockedbyldr_basicq_rf_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_basicq_rf', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_basicq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, flavor='rf')

condmeantime_blockedbyldr_q_rf_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_q_rf', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_q.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, flavor='rf')

condmeantime_blockedbyldr_noq_rf_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_noq_rf', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_noq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, flavor='rf')

# Support vector regression (svr)


mean_pct_blocked_by_ldr_basicq_svr_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_q_svr', 'obs', 'pct_blocked_by_ldr',
                          X_obs_basicq, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='svr')

mean_pct_blocked_by_ldr_q_svr_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_basicq_svr', 'obs', 'pct_blocked_by_ldr',
                          X_obs_q, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='svr')

mean_pct_blocked_by_ldr_noq_svr_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_noq_svr', 'obs', 'pct_blocked_by_ldr',
                          X_obs_noq, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='svr')

condmeantime_blockedbyldr_basicq_svr_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_basicq_svr', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_basicq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='svr')

condmeantime_blockedbyldr_q_svr_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_q_svr', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_q.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='svr')

condmeantime_blockedbyldr_noq_svr_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_noq_svr', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_noq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='svr')

# MLPRegressor Neural net (nn)


mean_pct_blocked_by_ldr_basicq_nn_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_q_nn', 'obs', 'pct_blocked_by_ldr',
                          X_obs_basicq, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='nn')

mean_pct_blocked_by_ldr_q_nn_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_basicq_nn', 'obs', 'pct_blocked_by_ldr',
                          X_obs_q, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='nn')

mean_pct_blocked_by_ldr_noq_nn_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_noq_nn', 'obs', 'pct_blocked_by_ldr',
                          X_obs_noq, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='nn')

condmeantime_blockedbyldr_basicq_nn_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_basicq_nn', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_basicq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='nn')

condmeantime_blockedbyldr_q_nn_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_q_nn', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_q.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='nn')

condmeantime_blockedbyldr_noq_nn_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_noq_nn', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_noq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='nn')

obs_results = {
                'mean_pct_blocked_by_ldr_basicq_lm_results': mean_pct_blocked_by_ldr_basicq_lm_results,
                'mean_pct_blocked_by_ldr_q_lm_results': mean_pct_blocked_by_ldr_q_lm_results,
                'mean_pct_blocked_by_ldr_noq_lm_results': mean_pct_blocked_by_ldr_noq_lm_results,
                'mean_pct_blocked_by_ldr_basicq_lassocv_results': mean_pct_blocked_by_ldr_basicq_lassocv_results,
                'mean_pct_blocked_by_ldr_q_lassocv_results': mean_pct_blocked_by_ldr_q_lassocv_results,
                'mean_pct_blocked_by_ldr_noq_lassocv_results': mean_pct_blocked_by_ldr_noq_lassocv_results,
                'mean_pct_blocked_by_ldr_basicq_poly_results': mean_pct_blocked_by_ldr_basicq_poly_results,
                'mean_pct_blocked_by_ldr_q_poly_results': mean_pct_blocked_by_ldr_q_poly_results,
                'mean_pct_blocked_by_ldr_noq_poly_results': mean_pct_blocked_by_ldr_noq_poly_results,
                'mean_pct_blocked_by_ldr_basicq_rf_results': mean_pct_blocked_by_ldr_basicq_rf_results,
                'mean_pct_blocked_by_ldr_q_rf_results': mean_pct_blocked_by_ldr_q_rf_results,
                'mean_pct_blocked_by_ldr_noq_rf_results': mean_pct_blocked_by_ldr_noq_rf_results,
                'mean_pct_blocked_by_ldr_basicq_svr_results': mean_pct_blocked_by_ldr_basicq_svr_results,
                'mean_pct_blocked_by_ldr_q_svr_results': mean_pct_blocked_by_ldr_q_svr_results,
                'mean_pct_blocked_by_ldr_noq_svr_results': mean_pct_blocked_by_ldr_noq_svr_results,
                'mean_pct_blocked_by_ldr_basicq_nn_results': mean_pct_blocked_by_ldr_basicq_nn_results,
                'mean_pct_blocked_by_ldr_q_nn_results': mean_pct_blocked_by_ldr_q_nn_results,
                'mean_pct_blocked_by_ldr_noq_nn_results': mean_pct_blocked_by_ldr_noq_nn_results,
                'condmeantime_blockedbyldr_q_basicq_lm_results': condmeantime_blockedbyldr_basicq_lm_results,
                'condmeantime_blockedbyldr_q_q_lm_results': condmeantime_blockedbyldr_q_lm_results,
                'condmeantime_blockedbyldr_q_noq_lm_results': condmeantime_blockedbyldr_noq_lm_results,
                'condmeantime_blockedbyldr_q_basicq_lassocv_results': condmeantime_blockedbyldr_basicq_lassocv_results,
                'condmeantime_blockedbyldr_q_q_lassocv_results': condmeantime_blockedbyldr_q_lassocv_results,
                'condmeantime_blockedbyldr_q_noq_lassocv_results': condmeantime_blockedbyldr_noq_lassocv_results,
                'condmeantime_blockedbyldr_q_basicq_poly_results': condmeantime_blockedbyldr_basicq_poly_results,
                'condmeantime_blockedbyldr_q_q_poly_results': condmeantime_blockedbyldr_q_poly_results,
                'condmeantime_blockedbyldr_q_noq_poly_results': condmeantime_blockedbyldr_noq_poly_results,
                'condmeantime_blockedbyldr_q_basicq_rf_results': condmeantime_blockedbyldr_basicq_rf_results,
                'condmeantime_blockedbyldr_q_q_rf_results': condmeantime_blockedbyldr_q_rf_results,
                'condmeantime_blockedbyldr_q_noq_rf_results': condmeantime_blockedbyldr_noq_rf_results,
                'condmeantime_blockedbyldr_q_basicq_svr_results': condmeantime_blockedbyldr_basicq_svr_results,
                'condmeantime_blockedbyldr_q_q_svr_results': condmeantime_blockedbyldr_q_svr_results,
                'condmeantime_blockedbyldr_q_noq_svr_results': condmeantime_blockedbyldr_noq_svr_results,
                'condmeantime_blockedbyldr_q_basicq_nn_results': condmeantime_blockedbyldr_basicq_nn_results,
                'condmeantime_blockedbyldr_q_nn_results': condmeantime_blockedbyldr_q_nn_results,
                'condmeantime_blockedbyldr_noq_nn_results': condmeantime_blockedbyldr_noq_nn_results

               }


# Pickle the results
with open(Path(output_path, f"obs_results2_{experiment}.pkl"), 'wb') as pickle_file:
    pickle.dump(obs_results, pickle_file)
