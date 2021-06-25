from pathlib import Path
import pickle

import pandas as pd

from mmfitting import crossval_summarize_mm

data_path = Path("data")
output_path = Path("output")
figures_path = Path("output", "figures")
raw_data_path = Path("data", "raw")

# X matrices
X_pp_noq = pd.read_csv(Path(data_path, 'X_pp_noq.csv'), index_col=0)
X_pp_basicq = pd.read_csv(Path(data_path, 'X_pp_basicq.csv'), index_col=0)

# y vectors
y_pp_occ_mean = pd.read_csv(Path(data_path, 'y_pp_occ_mean.csv'), index_col=0, squeeze=True)
y_pp_occ_p95 = pd.read_csv(Path(data_path, 'y_pp_occ_p95.csv'), index_col=0, squeeze=True)

# Fit models

## Linear regression (lm)
pp_occ_mean_basicq_lm_results = \
    crossval_summarize_mm('pp_occ_mean_basicq_lm', X_pp_basicq, y_pp_occ_mean, scale=False, flavor='lm')

pp_occ_mean_noq_lm_results = \
    crossval_summarize_mm('pp_occ_mean_noq_lm', X_pp_noq, y_pp_occ_mean, scale=False, flavor='lm')


pp_occ_p95_basicq_lm_results = \
    crossval_summarize_mm('pp_occ_p95_basicq_lm', X_pp_basicq, y_pp_occ_p95, scale=False, flavor='lm')

pp_occ_p95_noq_lm_results = \
    crossval_summarize_mm('pp_occ_p95_noq_lm', X_pp_noq, y_pp_occ_p95, scale=False, flavor='lm')

# LassoCV (lassocv)
pp_occ_mean_basicq_lassocv_results = \
    crossval_summarize_mm('pp_occ_mean_basicq_lassocv', X_pp_basicq, y_pp_occ_mean,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

pp_occ_mean_noq_lassocv_results = \
    crossval_summarize_mm('pp_occ_mean_noq_lassocv', X_pp_noq, y_pp_occ_mean,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

pp_occ_p95_basicq_lassocv_results = \
    crossval_summarize_mm('pp_occ_p95_basicq_lassocv', X_pp_basicq, y_pp_occ_p95,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

pp_occ_p95_noq_lassocv_results = \
    crossval_summarize_mm('pp_occ_p95_noq_lassocv', X_pp_noq, y_pp_occ_p95,
                          scale=True, flavor='lassocv', lasso_max_iter=3000)

# Polynomial regression (poly)
pp_occ_mean_basicq_poly_results = \
    crossval_summarize_mm('pp_occ_mean_basicq_poly', X_pp_basicq, y_pp_occ_mean, scale=False, flavor='poly')

pp_occ_mean_noq_poly_results = \
    crossval_summarize_mm('pp_occ_mean_noq_poly', X_pp_noq, y_pp_occ_mean, scale=False, flavor='poly')


pp_occ_p95_basicq_poly_results = \
    crossval_summarize_mm('pp_occ_p95_basicq_poly', X_pp_basicq, y_pp_occ_p95, scale=False, flavor='poly')

pp_occ_p95_noq_poly_results = \
    crossval_summarize_mm('pp_occ_p95_noq_poly', X_pp_noq, y_pp_occ_p95, scale=False, flavor='poly')

# Random forest (rf)
pp_occ_mean_basicq_rf_results = \
    crossval_summarize_mm('pp_occ_mean_basicq_rf', X_pp_basicq, y_pp_occ_mean, scale=False, flavor='rf')

pp_occ_mean_noq_rf_results = \
    crossval_summarize_mm('pp_occ_mean_noq_rf', X_pp_noq, y_pp_occ_mean, scale=False, flavor='rf')

pp_occ_p95_basicq_rf_results = \
    crossval_summarize_mm('pp_occ_p95_basicq_rf', X_pp_basicq, y_pp_occ_p95, scale=False, flavor='rf')

pp_occ_p95_noq_rf_results = \
    crossval_summarize_mm('pp_occ_p95_noq_rf', X_pp_noq, y_pp_occ_p95, scale=False, flavor='rf')

# Support vector regression (svr)
pp_occ_mean_basicq_svr_results = \
    crossval_summarize_mm('ldr_occ_mean_basicq_svr', X_pp_basicq, y_pp_occ_mean, flavor='svr', scale=True)

pp_occ_mean_noq_svr_results = \
    crossval_summarize_mm('ldr_occ_mean_noq_svr', X_pp_noq, y_pp_occ_mean, flavor='svr', scale=True)


pp_occ_p95_basicq_svr_results = \
    crossval_summarize_mm('ldr_occ_p95_basicq_svr', X_pp_basicq, y_pp_occ_p95, flavor='svr', scale=True)

pp_occ_p95_noq_svr_results = \
    crossval_summarize_mm('ldr_occ_p95_noq_svr', X_pp_noq, y_pp_occ_p95, flavor='svr', scale=True)

# MLPRegressor Neural net (nn)
pp_occ_mean_basicq_nn_results = \
    crossval_summarize_mm('pp_occ_mean_basicq_nn', X_pp_basicq, y_pp_occ_mean, flavor='nn', scale=True)

pp_occ_mean_noq_nn_results = \
    crossval_summarize_mm('pp_occ_mean_noq_nn', X_pp_noq, y_pp_occ_mean, flavor='nn', scale=True)

pp_occ_p95_basicq_nn_results = \
    crossval_summarize_mm('pp_occ_p95_basicq_nn', X_pp_basicq, y_pp_occ_p95, flavor='nn', scale=True)

pp_occ_p95_noq_nn_results = \
    crossval_summarize_mm('pp_occ_p95_noq_nn', X_pp_noq, y_pp_occ_p95, flavor='nn', scale=True)

# Gather results

pp_results = {'pp_occ_mean_basicq_lm_results': pp_occ_mean_basicq_lm_results,
              'pp_occ_mean_noq_lm_results': pp_occ_mean_noq_lm_results,
              'pp_occ_mean_basicq_lassocv_results': pp_occ_mean_basicq_lassocv_results,
              'pp_occ_mean_noq_lassocv_results': pp_occ_mean_noq_lassocv_results,
              'pp_occ_mean_basicq_poly_results': pp_occ_mean_basicq_poly_results,
              'pp_occ_mean_noq_poly_results': pp_occ_mean_noq_poly_results,
              'pp_occ_mean_basicq_rf_results': pp_occ_mean_basicq_rf_results,
              'pp_occ_mean_noq_rf_results': pp_occ_mean_noq_rf_results,
              'pp_occ_mean_basicq_svr_results': pp_occ_mean_basicq_svr_results,
              'pp_occ_mean_noq_svr_results': pp_occ_mean_noq_svr_results,
              'pp_occ_mean_basicq_nn_results': pp_occ_mean_basicq_nn_results,
              'pp_occ_mean_noq_nn_results': pp_occ_mean_noq_nn_results
              }


# Pickle the results
with open(Path(output_path, "pp_results.pkl"), 'wb') as pickle_file:
    pickle.dump(pp_results, pickle_file)