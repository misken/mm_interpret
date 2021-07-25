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
                'obs_occ_p95_noq_nn_results': obs_occ_p95_noq_nn_results

               }


# Pickle the results
with open(Path(output_path, f"obs_results1_{experiment}.pkl"), 'wb') as pickle_file:
    pickle.dump(obs_results, pickle_file)
