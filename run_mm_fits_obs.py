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

## Linear regression (lm)
obs_occ_mean_basicq_lm_results = \
    crossval_summarize_mm('obs_occ_mean_basicq_lm', 'obs', 'occ_mean',
                          X_obs_basicq, y_obs_occ_mean, scale=False, flavor='lm')

obs_occ_mean_q_lm_results = \
    crossval_summarize_mm('obs_occ_mean_q_lm', 'obs', 'occ_mean',
                          X_obs_q, y_obs_occ_mean, scale=False, flavor='lm')

obs_occ_mean_noq_lm_results = \
    crossval_summarize_mm('obs_occ_mean_noq_lm', 'obs', 'occ_mean',
                          X_obs_noq, y_obs_occ_mean, scale=False, flavor='lm')


obs_occ_p95_basicq_lm_results = \
    crossval_summarize_mm('obs_occ_p95_basicq_lm', 'obs', 'occ_p95',
                          X_obs_basicq, y_obs_occ_p95, scale=False, flavor='lm')

obs_occ_p95_q_lm_results = \
    crossval_summarize_mm('obs_occ_p95_q_lm', 'obs', 'occ_p95',
                          X_obs_q, y_obs_occ_p95, scale=False, flavor='lm')

obs_occ_p95_noq_lm_results = \
    crossval_summarize_mm('obs_occ_p95_noq_lm', 'obs', 'occ_p95',
                          X_obs_noq, y_obs_occ_p95, scale=False, flavor='lm')

mean_pct_blocked_by_ldr_basicq_lm_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_q_lm', 'obs', 'pct_blocked_by_ldr',
                          X_obs_q, y_mean_pct_blocked_by_ldr,
                                               scale=False, fit_intercept=True, flavor='lm')

mean_pct_blocked_by_ldr_q_lm_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_basicq_lm', 'obs', 'pct_blocked_by_ldr',
                          X_obs_basicq, y_mean_pct_blocked_by_ldr,
                                               scale=False, fit_intercept=True, flavor='lm')

mean_pct_blocked_by_ldr_noq_lm_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_noq_lm', 'obs', 'pct_blocked_by_ldr',
                          X_obs_noq, y_mean_pct_blocked_by_ldr,
                                               scale=False, fit_intercept=True, flavor='lm')

condmeantime_blockedbyldr_basicq_lm_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_basicq', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_basicq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, fit_intercept=True, flavor='lm')

condmeantime_blockedbyldr_q_lm_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_q', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_q.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, fit_intercept=True, flavor='lm')

condmeantime_blockedbyldr_noq_lm_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_noq', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_noq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, fit_intercept=True, flavor='lm')

# LassoCV (lassocv)
obs_occ_mean_basicq_lassocv_results = \
    crossval_summarize_mm('obs_occ_mean_basicq_lassocv', 'obs', 'occ_mean',
                          X_obs_basicq, y_obs_occ_mean,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

obs_occ_mean_q_lassocv_results = \
    crossval_summarize_mm('obs_occ_mean_q_lassocv', 'obs', 'occ_mean',
                          X_obs_q, y_obs_occ_mean,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

obs_occ_mean_noq_lassocv_results = \
    crossval_summarize_mm('obs_occ_mean_noq_lassocv', 'obs', 'occ_mean',
                          X_obs_noq, y_obs_occ_mean,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

obs_occ_p95_basicq_lassocv_results = \
    crossval_summarize_mm('obs_occ_p95_basicq_lassocv', 'obs', 'occ_p95',
                          X_obs_basicq, y_obs_occ_p95,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

obs_occ_p95_q_lassocv_results = \
    crossval_summarize_mm('obs_occ_p95_q_lassocv', 'obs', 'occ_p95',
                          X_obs_q, y_obs_occ_p95,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

obs_occ_p95_noq_lassocv_results = \
    crossval_summarize_mm('obs_occ_p95_noq_lassocv', 'obs', 'occ_p95',
                          X_obs_noq, y_obs_occ_p95,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

mean_pct_blocked_by_ldr_basicq_lassocv_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_q_lassocv', 'obs', 'pct_blocked_by_ldr',
                          X_obs_q, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)

mean_pct_blocked_by_ldr_q_lassocv_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_basicq_lassocv', 'obs', 'pct_blocked_by_ldr',
                          X_obs_basicq, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)

mean_pct_blocked_by_ldr_noq_lassocv_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_noq_lassocv', 'obs', 'pct_blocked_by_ldr',
                          X_obs_noq, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)


condmeantime_blockedbyldr_basicq_lassocv_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_basicq', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_basicq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)

condmeantime_blockedbyldr_q_lassocv_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_q', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_q.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)

condmeantime_blockedbyldr_noq_lassocv_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_noq', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_noq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='lassocv', lasso_max_iter=3000)


# Polynomial regression (poly)
obs_occ_mean_basicq_poly_results = \
    crossval_summarize_mm('obs_occ_mean_basicq_poly', 'obs', 'occ_mean',
                          X_obs_basicq, y_obs_occ_mean, scale=False, flavor='poly')

obs_occ_mean_q_poly_results = \
    crossval_summarize_mm('obs_occ_mean_q_poly', 'obs', 'occ_mean',
                          X_obs_q, y_obs_occ_mean, scale=False, flavor='poly')

obs_occ_mean_noq_poly_results = \
    crossval_summarize_mm('obs_occ_mean_noq_poly', 'obs', 'occ_mean',
                          X_obs_noq, y_obs_occ_mean, scale=False, flavor='poly')


obs_occ_p95_basicq_poly_results = \
    crossval_summarize_mm('obs_occ_p95_basicq_poly', 'obs', 'occ_p95',
                          X_obs_basicq, y_obs_occ_p95, scale=False, flavor='poly')

obs_occ_p95_q_poly_results = \
    crossval_summarize_mm('obs_occ_p95_q_poly', 'obs', 'occ_p95',
                          X_obs_q, y_obs_occ_p95, scale=False, flavor='poly')

obs_occ_p95_noq_poly_results = \
    crossval_summarize_mm('obs_occ_p95_noq_poly', 'obs', 'occ_p95',
                          X_obs_noq, y_obs_occ_p95, scale=False, flavor='poly')

mean_pct_blocked_by_ldr_basicq_poly_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_q_poly', 'obs', 'pct_blocked_by_ldr',
                          X_obs_q, y_mean_pct_blocked_by_ldr,
                                               scale=False, flavor='lm')

mean_pct_blocked_by_ldr_q_poly_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_basicq_poly', 'obs', 'pct_blocked_by_ldr',
                          X_obs_basicq, y_mean_pct_blocked_by_ldr,
                                               scale=False, flavor='lm')

mean_pct_blocked_by_ldr_noq_poly_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_noq_poly', 'obs', 'pct_blocked_by_ldr',
                          X_obs_noq, y_mean_pct_blocked_by_ldr,
                                               scale=False, flavor='lm')

condmeantime_blockedbyldr_basicq_poly_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_basicq', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_basicq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, flavor='lm')

condmeantime_blockedbyldr_q_poly_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_q', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_q.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, flavor='lm')

condmeantime_blockedbyldr_noq_poly_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_noq', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_noq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, flavor='lm')

# Random forest (rf)
obs_occ_mean_basicq_rf_results = \
    crossval_summarize_mm('obs_occ_mean_basicq_rf', 'obs', 'occ_mean',
                          X_obs_basicq, y_obs_occ_mean, scale=False, flavor='rf')

obs_occ_mean_q_rf_results = \
    crossval_summarize_mm('obs_occ_mean_q_rf', 'obs', 'occ_mean',
                          X_obs_q, y_obs_occ_mean, scale=False, flavor='rf')

obs_occ_mean_noq_rf_results = \
    crossval_summarize_mm('obs_occ_mean_noq_rf', 'obs', 'occ_mean',
                          X_obs_noq, y_obs_occ_mean, scale=False, flavor='rf')

obs_occ_p95_basicq_rf_results = \
    crossval_summarize_mm('obs_occ_p95_basicq_rf', 'obs', 'occ_p95',
                          X_obs_basicq, y_obs_occ_p95, scale=False, flavor='rf')

obs_occ_p95_q_rf_results = \
    crossval_summarize_mm('obs_occ_p95_q_rf', 'obs', 'occ_p95',
                          X_obs_q, y_obs_occ_p95, scale=False, flavor='rf')

obs_occ_p95_noq_rf_results = \
    crossval_summarize_mm('obs_occ_p95_noq_rf', 'obs', 'occ_p95',
                          X_obs_noq, y_obs_occ_p95, scale=False, flavor='rf')

mean_pct_blocked_by_ldr_basicq_rf_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_q_rf', 'obs', 'pct_blocked_by_ldr',
                          X_obs_q, y_mean_pct_blocked_by_ldr,
                                               scale=False, flavor='rf')

mean_pct_blocked_by_ldr_q_rf_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_basicq_rf', 'obs', 'pct_blocked_by_ldr',
                          X_obs_basicq, y_mean_pct_blocked_by_ldr,
                                               scale=False, flavor='rf')

mean_pct_blocked_by_ldr_noq_rf_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_noq_rf', 'obs', 'pct_blocked_by_ldr',
                          X_obs_noq, y_mean_pct_blocked_by_ldr,
                                               scale=False, flavor='rf')

condmeantime_blockedbyldr_basicq_rf_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_basicq', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_basicq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, flavor='rf')

condmeantime_blockedbyldr_q_rf_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_q', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_q.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, flavor='rf')

condmeantime_blockedbyldr_noq_rf_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_noq', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_noq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=False, flavor='rf')

# Support vector regression (svr)
obs_occ_mean_basicq_svr_results = \
    crossval_summarize_mm('obs_occ_mean_basicq_svr', 'obs', 'occ_mean',
                          X_obs_basicq, y_obs_occ_mean, flavor='svr', scale=True)

obs_occ_mean_q_svr_results = \
    crossval_summarize_mm('obs_occ_mean_q_svr', 'obs', 'occ_mean',
                          X_obs_q, y_obs_occ_mean, flavor='svr', scale=True)

obs_occ_mean_noq_svr_results = \
    crossval_summarize_mm('obs_occ_mean_noq_svr', 'obs', 'occ_mean',
                          X_obs_noq, y_obs_occ_mean, flavor='svr', scale=True)


obs_occ_p95_basicq_svr_results = \
    crossval_summarize_mm('obs_occ_p95_basicq_svr', 'obs', 'occ_p95',
                          X_obs_basicq, y_obs_occ_p95, flavor='svr', scale=True)

obs_occ_p95_q_svr_results = \
    crossval_summarize_mm('obs_occ_p95_q_svr', 'obs', 'occ_p95',
                          X_obs_q, y_obs_occ_p95, flavor='svr', scale=True)

obs_occ_p95_noq_svr_results = \
    crossval_summarize_mm('obs_occ_p95_noq_svr', 'obs', 'occ_p95',
                          X_obs_noq, y_obs_occ_p95, flavor='svr', scale=True)

mean_pct_blocked_by_ldr_basicq_svr_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_q_svr', 'obs', 'pct_blocked_by_ldr',
                          X_obs_q, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='svr')

mean_pct_blocked_by_ldr_q_svr_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_basicq_svr', 'obs', 'pct_blocked_by_ldr',
                          X_obs_basicq, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='svr')

mean_pct_blocked_by_ldr_noq_svr_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_noq_svr', 'obs', 'pct_blocked_by_ldr',
                          X_obs_noq, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='svr')

condmeantime_blockedbyldr_basicq_svr_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_basicq', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_basicq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='svr')

condmeantime_blockedbyldr_q_svr_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_q', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_q.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='svr')

condmeantime_blockedbyldr_noq_svr_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_noq', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_noq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='svr')

# MLPRegressor Neural net (nn)
obs_occ_mean_basicq_nn_results = \
    crossval_summarize_mm('obs_occ_mean_basicq_nn', 'obs', 'occ_mean',
                          X_obs_basicq, y_obs_occ_mean, flavor='nn', scale=True)

obs_occ_mean_q_nn_results = \
    crossval_summarize_mm('obs_occ_mean_q_nn', 'obs', 'occ_mean',
                          X_obs_q, y_obs_occ_mean, flavor='nn', scale=True)

obs_occ_mean_noq_nn_results = \
    crossval_summarize_mm('obs_occ_mean_noq_nn', 'obs', 'occ_mean',
                          X_obs_noq, y_obs_occ_mean, flavor='nn', scale=True)

obs_occ_p95_basicq_nn_results = \
    crossval_summarize_mm('obs_occ_p95_basicq_nn', 'obs', 'occ_p95',
                          X_obs_basicq, y_obs_occ_p95, flavor='nn', scale=True)

obs_occ_p95_q_nn_results = \
    crossval_summarize_mm('obs_occ_p95_q_nn', 'obs', 'occ_p95',
                          X_obs_q, y_obs_occ_p95, flavor='nn', scale=True)

obs_occ_p95_noq_nn_results = \
    crossval_summarize_mm('obs_occ_p95_noq_nn', 'obs', 'occ_p95',
                          X_obs_noq, y_obs_occ_p95, flavor='nn', scale=True)

mean_pct_blocked_by_ldr_basicq_nn_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_q_nn', 'obs', 'pct_blocked_by_ldr',
                          X_obs_q, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='nn')

mean_pct_blocked_by_ldr_q_nn_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_basicq_nn', 'obs', 'pct_blocked_by_ldr',
                          X_obs_basicq, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='nn')

mean_pct_blocked_by_ldr_noq_nn_results = \
    crossval_summarize_mm('mean_pct_blocked_by_ldr_noq_nn', 'obs', 'pct_blocked_by_ldr',
                          X_obs_noq, y_mean_pct_blocked_by_ldr,
                                               scale=True, flavor='nn')

condmeantime_blockedbyldr_basicq_nn_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_basicq', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_basicq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='nn')

condmeantime_blockedbyldr_q_nn_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_q', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_q.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='nn')

condmeantime_blockedbyldr_noq_nn_results = \
    crossval_summarize_mm('condmeantime_blockedbyldr_noq', 'obs', 'condmeantime_blockedbyldr',
                          X_obs_noq.iloc[:135], y_condmeantime_blockedbyldr.iloc[:135],
                                               scale=True, flavor='nn')

obs_results = {'obs_occ_mean_basicq_lm_results': obs_occ_mean_basicq_lm_results,
                'obs_occ_mean_q_lm_results': obs_occ_mean_q_lm_results,
                'obs_occ_mean_noq_lm_results': obs_occ_mean_noq_lm_results,
                'obs_occ_mean_basicq_lassocv_results': obs_occ_mean_basicq_lassocv_results,
                'obs_occ_mean_q_lassocv_results': obs_occ_mean_q_lassocv_results,
                'obs_occ_mean_noq_lassocv_results': obs_occ_mean_noq_lassocv_results,
                'obs_occ_mean_basicq_poly_results': obs_occ_mean_basicq_poly_results,
                'obs_occ_mean_q_poly_results': obs_occ_mean_q_poly_results,
                'obs_occ_mean_noq_poly_results': obs_occ_mean_noq_poly_results,
                'obs_occ_mean_basicq_rf_results': obs_occ_mean_basicq_rf_results,
                'obs_occ_mean_q_rf_results': obs_occ_mean_q_rf_results,
                'obs_occ_mean_noq_rf_results': obs_occ_mean_noq_rf_results,
                'obs_occ_mean_basicq_svr_results': obs_occ_mean_basicq_svr_results,
                'obs_occ_mean_q_svr_results': obs_occ_mean_q_svr_results,
                'obs_occ_mean_noq_svr_results': obs_occ_mean_noq_svr_results,
                'obs_occ_mean_basicq_nn_results': obs_occ_mean_basicq_nn_results,
                'obs_occ_mean_q_nn_results': obs_occ_mean_q_nn_results,
                'obs_occ_mean_noq_nn_results': obs_occ_mean_noq_nn_results,
                'obs_occ_p95_basicq_lm_results': obs_occ_p95_basicq_lm_results,
                'obs_occ_p95_q_lm_results': obs_occ_p95_q_lm_results,
                'obs_occ_p95_noq_lm_results': obs_occ_p95_noq_lm_results,
                'obs_occ_p95_basicq_lassocv_results': obs_occ_p95_basicq_lassocv_results,
                'obs_occ_p95_q_lassocv_results': obs_occ_p95_q_lassocv_results,
                'obs_occ_p95_noq_lassocv_results': obs_occ_p95_noq_lassocv_results,
                'obs_occ_p95_basicq_poly_results': obs_occ_p95_basicq_poly_results,
                'obs_occ_p95_q_poly_results': obs_occ_p95_q_poly_results,
                'obs_occ_p95_noq_poly_results': obs_occ_p95_noq_poly_results,
                'obs_occ_p95_basicq_rf_results': obs_occ_p95_basicq_rf_results,
                'obs_occ_p95_q_rf_results': obs_occ_p95_q_rf_results,
                'obs_occ_p95_noq_rf_results': obs_occ_p95_noq_rf_results,
                'obs_occ_p95_basicq_svr_results': obs_occ_p95_basicq_svr_results,
                'obs_occ_p95_q_svr_results': obs_occ_p95_q_svr_results,
                'obs_occ_p95_noq_svr_results': obs_occ_p95_noq_svr_results,
                'obs_occ_p95_basicq_nn_results': obs_occ_p95_basicq_nn_results,
                'obs_occ_p95_q_nn_results': obs_occ_p95_q_nn_results,
                'obs_occ_p95_noq_nn_results': obs_occ_p95_noq_nn_results,
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
with open(Path(output_path, "obs_results.pkl"), 'wb') as pickle_file:
    pickle.dump(obs_results, pickle_file)
