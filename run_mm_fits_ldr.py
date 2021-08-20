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
pickle_filename = f"ldr_results_{experiment}.pkl"

# X matrices
X_ldr_noq = pd.read_csv(Path(data_path, f'X_ldr_noq_{experiment}.csv'), index_col=0)
X_ldr_basicq = pd.read_csv(Path(data_path, f'X_ldr_basicq_{experiment}.csv'), index_col=0)
X_ldr_q = pd.read_csv(Path(data_path, f'X_ldr_q_{experiment}.csv'), index_col=0)

# y vectors
y_ldr_occ_mean = pd.read_csv(Path(data_path, f'y_ldr_occ_mean_{experiment}.csv'), index_col=0, squeeze=True)
y_ldr_occ_p95 = pd.read_csv(Path(data_path, f'y_ldr_occ_p95_{experiment}.csv'), index_col=0, squeeze=True)
y_mean_pct_blocked_by_pp = pd.read_csv(Path(data_path, f'y_mean_pct_blocked_by_pp_{experiment}.csv'), index_col=0, squeeze=True)
y_condmeantime_blockedbypp = pd.read_csv(Path(data_path, f'y_condmeantime_blockedbypp_{experiment}.csv'), index_col=0, squeeze=True)


# Fit models

## Linear regression (lm)
ldr_occ_mean_basicq_lm_results = \
    crossval_summarize_mm('ldr_occ_mean_basicq_lm', 'ldr', 'occ_mean', X_ldr_basicq, y_ldr_occ_mean, scale=False, flavor='lm')

ldr_occ_mean_q_lm_results = \
    crossval_summarize_mm('ldr_occ_mean_q_lm', 'ldr', 'occ_mean', X_ldr_q, y_ldr_occ_mean, scale=False, flavor='lm')

ldr_occ_mean_noq_lm_results = \
    crossval_summarize_mm('ldr_occ_mean_noq_lm', 'ldr', 'occ_mean', X_ldr_noq, y_ldr_occ_mean, scale=False, flavor='lm')


ldr_occ_p95_basicq_lm_results = \
    crossval_summarize_mm('ldr_occ_p95_basicq_lm', 'ldr', 'occ_p95', X_ldr_basicq, y_ldr_occ_p95, scale=False, flavor='lm')

ldr_occ_p95_q_lm_results = \
    crossval_summarize_mm('ldr_occ_p95_q_lm', 'ldr', 'occ_p95', X_ldr_q, y_ldr_occ_p95, scale=False, flavor='lm')

ldr_occ_p95_noq_lm_results = \
    crossval_summarize_mm('ldr_occ_p95_noq_lm', 'ldr', 'occ_p95', X_ldr_noq, y_ldr_occ_p95, scale=False, flavor='lm')

mean_pct_blocked_by_pp_basicq_lm_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_q_lm', 'ldr', 'pct_blocked_by_pp', X_ldr_basicq, y_mean_pct_blocked_by_pp,
                                               scale=False, fit_intercept=True, flavor='lm')

mean_pct_blocked_by_pp_q_lm_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_basicq_lm', 'ldr', 'pct_blocked_by_pp', X_ldr_q, y_mean_pct_blocked_by_pp,
                                               scale=False, fit_intercept=True, flavor='lm')

mean_pct_blocked_by_pp_noq_lm_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_noq_lm', 'ldr', 'pct_blocked_by_pp', X_ldr_noq, y_mean_pct_blocked_by_pp,
                                               scale=False, fit_intercept=True, flavor='lm')

condmeantime_blockedbypp_basicq_lm_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_basicq_lm', 'ldr', 'condmeantime_blockedbypp', X_ldr_basicq.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=False, fit_intercept=True, flavor='lm')

condmeantime_blockedbypp_q_lm_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_q_lm', 'ldr', 'condmeantime_blockedbypp', X_ldr_q.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=False, fit_intercept=True, flavor='lm')

condmeantime_blockedbypp_noq_lm_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_noq_lm', 'ldr', 'condmeantime_blockedbypp', X_ldr_noq.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=False, fit_intercept=True, flavor='lm')

# LassoCV (lassocv)
ldr_occ_mean_basicq_lassocv_results = \
    crossval_summarize_mm('ldr_occ_mean_basicq_lassocv', 'ldr', 'occ_mean', X_ldr_basicq, y_ldr_occ_mean,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

ldr_occ_mean_q_lassocv_results = \
    crossval_summarize_mm('ldr_occ_mean_q_lassocv', 'ldr', 'occ_mean', X_ldr_q, y_ldr_occ_mean,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

ldr_occ_mean_noq_lassocv_results = \
    crossval_summarize_mm('ldr_occ_mean_noq_lassocv', 'ldr', 'occ_mean', X_ldr_noq, y_ldr_occ_mean,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

ldr_occ_p95_basicq_lassocv_results = \
    crossval_summarize_mm('ldr_occ_p95_basicq_lassocv', 'ldr', 'occ_p95', X_ldr_basicq, y_ldr_occ_p95,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

ldr_occ_p95_q_lassocv_results = \
    crossval_summarize_mm('ldr_occ_p95_q_lassocv', 'ldr', 'occ_p95', X_ldr_q, y_ldr_occ_p95,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

ldr_occ_p95_noq_lassocv_results = \
    crossval_summarize_mm('ldr_occ_p95_noq_lassocv', 'ldr', 'occ_p95', X_ldr_noq, y_ldr_occ_p95,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

mean_pct_blocked_by_pp_basicq_lassocv_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_q_lassocv', 'ldr', 'pct_blocked_by_pp', X_ldr_basicq, y_mean_pct_blocked_by_pp,
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)

mean_pct_blocked_by_pp_q_lassocv_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_basicq_lassocv', 'ldr', 'pct_blocked_by_pp', X_ldr_q, y_mean_pct_blocked_by_pp,
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)

mean_pct_blocked_by_pp_noq_lassocv_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_noq_lassocv', 'ldr', 'pct_blocked_by_pp', X_ldr_noq, y_mean_pct_blocked_by_pp,
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)


condmeantime_blockedbypp_basicq_lassocv_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_basicq_lassocv', 'ldr', 'condmeantime_blockedbypp', X_ldr_basicq.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)

condmeantime_blockedbypp_q_lassocv_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_q_lassocv', 'ldr', 'condmeantime_blockedbypp', X_ldr_q.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)

condmeantime_blockedbypp_noq_lassocv_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_noq_lassocv', 'ldr', 'condmeantime_blockedbypp', X_ldr_noq.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)


# Polynomial regression (poly)
ldr_occ_mean_basicq_poly_results = \
    crossval_summarize_mm('ldr_occ_mean_basicq_poly', 'ldr', 'occ_mean', X_ldr_basicq, y_ldr_occ_mean, scale=False, flavor='poly')

ldr_occ_mean_q_poly_results = \
    crossval_summarize_mm('ldr_occ_mean_q_poly', 'ldr', 'occ_mean', X_ldr_q, y_ldr_occ_mean, scale=False, flavor='poly')

ldr_occ_mean_noq_poly_results = \
    crossval_summarize_mm('ldr_occ_mean_noq_poly', 'ldr', 'occ_mean', X_ldr_noq, y_ldr_occ_mean, scale=False,
                          flavor='poly')


ldr_occ_p95_basicq_poly_results = \
    crossval_summarize_mm('ldr_occ_p95_basicq_poly', 'ldr', 'occ_p95', X_ldr_basicq, y_ldr_occ_p95, scale=False, flavor='poly')

ldr_occ_p95_q_poly_results = \
    crossval_summarize_mm('ldr_occ_p95_q_poly', 'ldr', 'occ_p95', X_ldr_q, y_ldr_occ_p95, scale=False, flavor='poly')

ldr_occ_p95_noq_poly_results = \
    crossval_summarize_mm('ldr_occ_p95_noq_poly', 'ldr', 'occ_p95', X_ldr_noq, y_ldr_occ_p95, scale=False, flavor='poly')

mean_pct_blocked_by_pp_basicq_poly_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_q_poly', 'ldr', 'pct_blocked_by_pp', X_ldr_basicq, y_mean_pct_blocked_by_pp,
                                               scale=False, flavor='lm')

mean_pct_blocked_by_pp_q_poly_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_basicq_poly', 'ldr', 'pct_blocked_by_pp', X_ldr_q, y_mean_pct_blocked_by_pp,
                                               scale=False, flavor='lm')

mean_pct_blocked_by_pp_noq_poly_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_noq_poly', 'ldr', 'pct_blocked_by_pp', X_ldr_noq, y_mean_pct_blocked_by_pp,
                                               scale=False, flavor='lm')

condmeantime_blockedbypp_basicq_poly_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_basicq_poly', 'ldr', 'condmeantime_blockedbypp', X_ldr_basicq.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=False, flavor='lm')

condmeantime_blockedbypp_q_poly_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_q_poly', 'ldr', 'condmeantime_blockedbypp', X_ldr_q.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=False, flavor='lm')

condmeantime_blockedbypp_noq_poly_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_noq_poly', 'ldr', 'condmeantime_blockedbypp', X_ldr_noq.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=False, flavor='lm')

# Random forest (rf)
ldr_occ_mean_basicq_rf_results = \
    crossval_summarize_mm('ldr_occ_mean_basicq_rf', 'ldr', 'occ_mean', X_ldr_basicq, y_ldr_occ_mean, scale=False, flavor='rf')

ldr_occ_mean_q_rf_results = \
    crossval_summarize_mm('ldr_occ_mean_q_rf', 'ldr', 'occ_mean', X_ldr_q, y_ldr_occ_mean, scale=False, flavor='rf')

ldr_occ_mean_noq_rf_results = \
    crossval_summarize_mm('ldr_occ_mean_noq_rf', 'ldr', 'occ_mean', X_ldr_noq, y_ldr_occ_mean, scale=False, flavor='rf')

ldr_occ_p95_basicq_rf_results = \
    crossval_summarize_mm('ldr_occ_p95_basicq_rf', 'ldr', 'occ_p95', X_ldr_basicq, y_ldr_occ_p95, scale=False, flavor='rf')

ldr_occ_p95_q_rf_results = \
    crossval_summarize_mm('ldr_occ_p95_q_rf', 'ldr', 'occ_p95', X_ldr_q, y_ldr_occ_p95, scale=False, flavor='rf')

ldr_occ_p95_noq_rf_results = \
    crossval_summarize_mm('ldr_occ_p95_noq_rf', 'ldr', 'occ_p95', X_ldr_noq, y_ldr_occ_p95, scale=False, flavor='rf')

mean_pct_blocked_by_pp_basicq_rf_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_q_rf', 'ldr', 'pct_blocked_by_pp', X_ldr_basicq, y_mean_pct_blocked_by_pp,
                                               scale=False, flavor='rf')

mean_pct_blocked_by_pp_q_rf_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_basicq_rf', 'ldr', 'pct_blocked_by_pp', X_ldr_q, y_mean_pct_blocked_by_pp,
                                               scale=False, flavor='rf')

mean_pct_blocked_by_pp_noq_rf_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_noq_rf', 'ldr', 'pct_blocked_by_pp', X_ldr_noq, y_mean_pct_blocked_by_pp,
                                               scale=False, flavor='rf')

condmeantime_blockedbypp_basicq_rf_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_basicq_rf', 'ldr', 'condmeantime_blockedbypp', X_ldr_basicq.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=False, flavor='rf')

condmeantime_blockedbypp_q_rf_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_q_rf', 'ldr', 'condmeantime_blockedbypp', X_ldr_q.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=False, flavor='rf')

condmeantime_blockedbypp_noq_rf_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_noq_rf', 'ldr', 'condmeantime_blockedbypp', X_ldr_noq.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=False, flavor='rf')

# Support vector regression (svr)
ldr_occ_mean_basicq_svr_results = \
    crossval_summarize_mm('ldr_occ_mean_basicq_svr', 'ldr', 'occ_mean', X_ldr_basicq, y_ldr_occ_mean, flavor='svr', scale=True)

ldr_occ_mean_q_svr_results = \
    crossval_summarize_mm('ldr_occ_mean_q_svr', 'ldr', 'occ_mean', X_ldr_q, y_ldr_occ_mean, flavor='svr', scale=True)

ldr_occ_mean_noq_svr_results = \
    crossval_summarize_mm('ldr_occ_mean_noq_svr', 'ldr', 'occ_mean', X_ldr_noq, y_ldr_occ_mean, flavor='svr', scale=True)


ldr_occ_p95_basicq_svr_results = \
    crossval_summarize_mm('ldr_occ_p95_basicq_svr', 'ldr', 'occ_p95', X_ldr_basicq, y_ldr_occ_p95, flavor='svr', scale=True)

ldr_occ_p95_q_svr_results = \
    crossval_summarize_mm('ldr_occ_p95_q_svr', 'ldr', 'occ_p95', X_ldr_q, y_ldr_occ_p95, flavor='svr', scale=True)

ldr_occ_p95_noq_svr_results = \
    crossval_summarize_mm('ldr_occ_p95_noq_svr', 'ldr', 'occ_p95', X_ldr_noq, y_ldr_occ_p95, flavor='svr', scale=True)

mean_pct_blocked_by_pp_basicq_svr_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_q_svr', 'ldr', 'pct_blocked_by_pp', X_ldr_basicq, y_mean_pct_blocked_by_pp,
                                               scale=True, flavor='svr')

mean_pct_blocked_by_pp_q_svr_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_basicq_svr', 'ldr', 'pct_blocked_by_pp', X_ldr_q, y_mean_pct_blocked_by_pp,
                                               scale=True, flavor='svr')

mean_pct_blocked_by_pp_noq_svr_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_noq_svr', 'ldr', 'pct_blocked_by_pp', X_ldr_noq, y_mean_pct_blocked_by_pp,
                                               scale=True, flavor='svr')

condmeantime_blockedbypp_basicq_svr_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_basicq_svr', 'ldr', 'condmeantime_blockedbypp', X_ldr_basicq.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=True, flavor='svr')

condmeantime_blockedbypp_q_svr_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_q_svr', 'ldr', 'condmeantime_blockedbyldr', X_ldr_q.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=True, flavor='svr')

condmeantime_blockedbypp_noq_svr_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_noq_svr', 'ldr', 'condmeantime_blockedbypp', X_ldr_noq.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=True, flavor='svr')

# MLPRegressor Neural net (nn)
ldr_occ_mean_basicq_nn_results = \
    crossval_summarize_mm('ldr_occ_mean_basicq_nn', 'ldr', 'occ_mean', X_ldr_basicq, y_ldr_occ_mean, flavor='nn', scale=True)

ldr_occ_mean_q_nn_results = \
    crossval_summarize_mm('ldr_occ_mean_q_nn', 'ldr', 'occ_mean', X_ldr_q, y_ldr_occ_mean, flavor='nn', scale=True)

ldr_occ_mean_noq_nn_results = \
    crossval_summarize_mm('ldr_occ_mean_noq_nn', 'ldr', 'occ_mean', X_ldr_noq, y_ldr_occ_mean, flavor='nn', scale=True)

ldr_occ_p95_basicq_nn_results = \
    crossval_summarize_mm('ldr_occ_p95_basicq_nn', 'ldr', 'occ_p95', X_ldr_basicq, y_ldr_occ_p95, flavor='nn', scale=True)

ldr_occ_p95_q_nn_results = \
    crossval_summarize_mm('ldr_occ_p95_q_nn', 'ldr', 'occ_p95', X_ldr_q, y_ldr_occ_p95, flavor='nn', scale=True)

ldr_occ_p95_noq_nn_results = \
    crossval_summarize_mm('ldr_occ_p95_noq_nn', 'ldr', 'occ_p95', X_ldr_noq, y_ldr_occ_p95, flavor='nn', scale=True)

mean_pct_blocked_by_pp_basicq_nn_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_q_nn', 'ldr', 'pct_blocked_by_pp', X_ldr_basicq, y_mean_pct_blocked_by_pp,
                                               scale=True, flavor='nn')

mean_pct_blocked_by_pp_q_nn_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_basicq_nn', 'ldr', 'pct_blocked_by_pp', X_ldr_q, y_mean_pct_blocked_by_pp,
                                               scale=True, flavor='nn')

mean_pct_blocked_by_pp_noq_nn_results = \
    crossval_summarize_mm('mean_pct_blocked_by_pp_noq_nn', 'ldr', 'pct_blocked_by_pp', X_ldr_noq, y_mean_pct_blocked_by_pp,
                                               scale=True, flavor='nn')

condmeantime_blockedbypp_basicq_nn_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_basicq_nn', 'ldr', 'condmeantime_blockedbypp', X_ldr_basicq.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=True, flavor='nn')

condmeantime_blockedbypp_q_nn_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_q_nn', 'ldr', 'condmeantime_blockedbypp', X_ldr_q.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=True, flavor='nn')

condmeantime_blockedbypp_noq_nn_results = \
    crossval_summarize_mm('condmeantime_blockedbypp_noq_nn', 'ldr', 'condmeantime_blockedbypp', X_ldr_noq.iloc[:135], y_condmeantime_blockedbypp.iloc[:135],
                                               scale=True, flavor='nn')

# Queueing models
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
                          flavor='erlangc', col_idx_arate=0, col_idx_meansvctime=4, col_idx_numservers=6)


ldr_results = {'ldr_occ_mean_basicq_lm_results': ldr_occ_mean_basicq_lm_results,
                'ldr_occ_mean_q_lm_results': ldr_occ_mean_q_lm_results,
                'ldr_occ_mean_noq_lm_results': ldr_occ_mean_noq_lm_results,
                'ldr_occ_mean_basicq_lassocv_results': ldr_occ_mean_basicq_lassocv_results,
                'ldr_occ_mean_q_lassocv_results': ldr_occ_mean_q_lassocv_results,
                'ldr_occ_mean_noq_lassocv_results': ldr_occ_mean_noq_lassocv_results,
                'ldr_occ_mean_basicq_poly_results': ldr_occ_mean_basicq_poly_results,
                'ldr_occ_mean_q_poly_results': ldr_occ_mean_q_poly_results,
                'ldr_occ_mean_noq_poly_results': ldr_occ_mean_noq_poly_results,
                'ldr_occ_mean_basicq_rf_results': ldr_occ_mean_basicq_rf_results,
                'ldr_occ_mean_q_rf_results': ldr_occ_mean_q_rf_results,
                'ldr_occ_mean_noq_rf_results': ldr_occ_mean_noq_rf_results,
                'ldr_occ_mean_basicq_svr_results': ldr_occ_mean_basicq_svr_results,
                'ldr_occ_mean_q_svr_results': ldr_occ_mean_q_svr_results,
                'ldr_occ_mean_noq_svr_results': ldr_occ_mean_noq_svr_results,
                'ldr_occ_mean_basicq_nn_results': ldr_occ_mean_basicq_nn_results,
                'ldr_occ_mean_q_nn_results': ldr_occ_mean_q_nn_results,
                'ldr_occ_mean_noq_nn_results': ldr_occ_mean_noq_nn_results,
                'ldr_occ_mean_q_load_results': ldr_occ_mean_q_load_results,
                'ldr_occ_p95_basicq_lm_results': ldr_occ_p95_basicq_lm_results,
                'ldr_occ_p95_q_lm_results': ldr_occ_p95_q_lm_results,
                'ldr_occ_p95_noq_lm_results': ldr_occ_p95_noq_lm_results,
                'ldr_occ_p95_basicq_lassocv_results': ldr_occ_p95_basicq_lassocv_results,
                'ldr_occ_p95_q_lassocv_results': ldr_occ_p95_q_lassocv_results,
                'ldr_occ_p95_noq_lassocv_results': ldr_occ_p95_noq_lassocv_results,
                'ldr_occ_p95_basicq_poly_results': ldr_occ_p95_basicq_poly_results,
                'ldr_occ_p95_q_poly_results': ldr_occ_p95_q_poly_results,
                'ldr_occ_p95_noq_poly_results': ldr_occ_p95_noq_poly_results,
                'ldr_occ_p95_basicq_rf_results': ldr_occ_p95_basicq_rf_results,
                'ldr_occ_p95_q_rf_results': ldr_occ_p95_q_rf_results,
                'ldr_occ_p95_noq_rf_results': ldr_occ_p95_noq_rf_results,
                'ldr_occ_p95_basicq_svr_results': ldr_occ_p95_basicq_svr_results,
                'ldr_occ_p95_q_svr_results': ldr_occ_p95_q_svr_results,
                'ldr_occ_p95_noq_svr_results': ldr_occ_p95_noq_svr_results,
                'ldr_occ_p95_basicq_nn_results': ldr_occ_p95_basicq_nn_results,
                'ldr_occ_p95_q_nn_results': ldr_occ_p95_q_nn_results,
                'ldr_occ_p95_noq_nn_results': ldr_occ_p95_noq_nn_results,
                'ldr_occ_p95_q_sqrtload_results': ldr_occ_p95_q_sqrtload_results,
                'mean_pct_blocked_by_pp_basicq_lm_results': mean_pct_blocked_by_pp_basicq_lm_results,
                'mean_pct_blocked_by_pp_q_lm_results': mean_pct_blocked_by_pp_q_lm_results,
                'mean_pct_blocked_by_pp_noq_lm_results': mean_pct_blocked_by_pp_noq_lm_results,
                'mean_pct_blocked_by_pp_basicq_lassocv_results': mean_pct_blocked_by_pp_basicq_lassocv_results,
                'mean_pct_blocked_by_pp_q_lassocv_results': mean_pct_blocked_by_pp_q_lassocv_results,
                'mean_pct_blocked_by_pp_noq_lassocv_results': mean_pct_blocked_by_pp_noq_lassocv_results,
                'mean_pct_blocked_by_pp_basicq_poly_results': mean_pct_blocked_by_pp_basicq_poly_results,
                'mean_pct_blocked_by_pp_q_poly_results': mean_pct_blocked_by_pp_q_poly_results,
                'mean_pct_blocked_by_pp_noq_poly_results': mean_pct_blocked_by_pp_noq_poly_results,
                'mean_pct_blocked_by_pp_basicq_rf_results': mean_pct_blocked_by_pp_basicq_rf_results,
                'mean_pct_blocked_by_pp_q_rf_results': mean_pct_blocked_by_pp_q_rf_results,
                'mean_pct_blocked_by_pp_noq_rf_results': mean_pct_blocked_by_pp_noq_rf_results,
                'mean_pct_blocked_by_pp_basicq_svr_results': mean_pct_blocked_by_pp_basicq_svr_results,
                'mean_pct_blocked_by_pp_q_svr_results': mean_pct_blocked_by_pp_q_svr_results,
                'mean_pct_blocked_by_pp_noq_svr_results': mean_pct_blocked_by_pp_noq_svr_results,
                'mean_pct_blocked_by_pp_basicq_nn_results': mean_pct_blocked_by_pp_basicq_nn_results,
                'mean_pct_blocked_by_pp_q_nn_results': mean_pct_blocked_by_pp_q_nn_results,
                'mean_pct_blocked_by_pp_noq_nn_results': mean_pct_blocked_by_pp_noq_nn_results,
                'mean_pct_blocked_by_pp_q_erlangc_results': mean_pct_blocked_by_pp_q_erlangc_results,
                'condmeantime_blockedbypp_q_basicq_lm_results': condmeantime_blockedbypp_basicq_lm_results,
                'condmeantime_blockedbypp_q_q_lm_results': condmeantime_blockedbypp_q_lm_results,
                'condmeantime_blockedbypp_q_noq_lm_results': condmeantime_blockedbypp_noq_lm_results,
                'condmeantime_blockedbypp_q_basicq_lassocv_results': condmeantime_blockedbypp_basicq_lassocv_results,
                'condmeantime_blockedbypp_q_q_lassocv_results': condmeantime_blockedbypp_q_lassocv_results,
                'condmeantime_blockedbypp_q_noq_lassocv_results': condmeantime_blockedbypp_noq_lassocv_results,
                'condmeantime_blockedbypp_q_basicq_poly_results': condmeantime_blockedbypp_basicq_poly_results,
                'condmeantime_blockedbypp_q_q_poly_results': condmeantime_blockedbypp_q_poly_results,
                'condmeantime_blockedbypp_q_noq_poly_results': condmeantime_blockedbypp_noq_poly_results,
                'condmeantime_blockedbypp_q_basicq_rf_results': condmeantime_blockedbypp_basicq_rf_results,
                'condmeantime_blockedbypp_q_q_rf_results': condmeantime_blockedbypp_q_rf_results,
                'condmeantime_blockedbypp_q_noq_rf_results': condmeantime_blockedbypp_noq_rf_results,
                'condmeantime_blockedbypp_q_basicq_svr_results': condmeantime_blockedbypp_basicq_svr_results,
                'condmeantime_blockedbypp_q_q_svr_results': condmeantime_blockedbypp_q_svr_results,
                'condmeantime_blockedbypp_q_noq_svr_results': condmeantime_blockedbypp_noq_svr_results,
                'condmeantime_blockedbypp_q_basicq_nn_results': condmeantime_blockedbypp_basicq_nn_results,
                'condmeantime_blockedbypp_q_nn_results': condmeantime_blockedbypp_q_nn_results,
                'condmeantime_blockedbypp_noq_nn_results': condmeantime_blockedbypp_noq_nn_results

               }


# Pickle the results
with open(Path(output_path, pickle_filename), 'wb') as pickle_file:
    pickle.dump(ldr_results, pickle_file)
