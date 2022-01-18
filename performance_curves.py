import math
from pathlib import Path
import pickle
import itertools

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

from obnetwork import obnetwork
from qng import qng


def make_x_scenarios(scenarios_df, exp, data_path):
    """
    Generate performance curve dataframes consistent with the X matrices used in mm fitting

    Parameters
    ----------
    scenarios_df : pandas Dataframe

    exp : str
        Experiment id from original simuation runs from which metamodels were fitted
    data_path
        Location of X matrices files. Need to get the column specs from them so that we can create performance
        curve dataframes that can be used to generate predictions from previously fitted metamodels.

    Returns
    -------

    """

    meta_input_cols = ['arrival_rate', 'mean_los_obs', 'num_erlang_stages_obs',
                       'mean_los_ldr', 'num_erlang_stages_ldr', 'mean_los_csect', 'num_erlang_stages_csect',
                       'mean_los_pp_noc', 'mean_los_pp_c', 'num_erlang_stages_pp', 'c_sect_prob',
                       'mean_los_pp', 'cap_obs', 'cap_ldr', 'cap_pp']


    # Read X matrices
    X_ldr_q = pd.read_csv(Path(data_path, f'X_ldr_q_{exp}.csv'), index_col=0)
    X_ldr_q_cols = X_ldr_q.columns.tolist()

    X_ldr_noq = pd.read_csv(Path(data_path, f'X_ldr_noq_{exp}.csv'), index_col=0)
    X_ldr_noq_cols = X_ldr_noq.columns.tolist()

    X_ldr_occmean_onlyq = pd.read_csv(Path(data_path, f'X_ldr_occmean_onlyq_exp11.csv'), index_col=0)
    X_ldr_occmean_onlyq_cols = X_ldr_occmean_onlyq.columns.tolist()

    X_ldr_occp95_onlyq = pd.read_csv(Path(data_path, f'X_ldr_occp95_onlyq_exp11.csv'), index_col=0)
    X_ldr_occp95_onlyq_cols = X_ldr_occp95_onlyq.columns.tolist()

    X_ldr_prob_blockedby_pp_onlyq = pd.read_csv(Path(data_path, f'X_ldr_prob_blockedby_pp_onlyq_exp11.csv'), index_col=0)
    X_ldr_prob_blockedby_pp_onlyq_cols = X_ldr_prob_blockedby_pp_onlyq.columns.tolist()

    X_ldr_condmeantime_blockedby_pp_onlyq = \
        pd.read_csv(Path(data_path, f'X_ldr_condmeantime_blockedby_pp_onlyq_exp11.csv'), index_col=0)
    X_ldr_condmeantime_blockedby_pp_onlyq_cols = X_ldr_condmeantime_blockedby_pp_onlyq.columns.tolist()

    X_pp_basicq = pd.read_csv(Path(data_path, f'X_pp_basicq_{exp}.csv'), index_col=0)
    X_pp_basicq_cols = X_pp_basicq.columns.tolist()

    X_pp_noq = pd.read_csv(Path(data_path, f'X_pp_noq_{exp}.csv'), index_col=0)
    X_pp_noq_cols = X_pp_noq.columns.tolist()

    # Compute overall mean los and cv2 for PP and LDR
    scenarios_df['mean_los_pp'] = scenarios_df['c_sect_prob'] * scenarios_df['mean_los_pp_c'] + (
            1 - scenarios_df['c_sect_prob']) * scenarios_df['mean_los_pp_noc']

    scenarios_df['pp_cv2_svctime'] = scenarios_df.apply(lambda x: hyper_erlang_cv2(
        [x.mean_los_pp_noc, x.mean_los_pp_c], [x.num_erlang_stages_pp, x.num_erlang_stages_pp],
        [1 - x.c_sect_prob, x.c_sect_prob]), axis=1)

    scenarios_df['ldr_cv2_svctime'] = scenarios_df.apply(lambda x: 1 / x.num_erlang_stages_ldr, axis=1)

    # Compute load and rho related terms
    scenarios_df['load_ldr'] = scenarios_df['arrival_rate'] * scenarios_df['mean_los_ldr']
    scenarios_df['rho_ldr'] = scenarios_df['load_ldr'] / scenarios_df['cap_ldr']

    scenarios_df['load_pp'] = scenarios_df['arrival_rate'] * scenarios_df['mean_los_pp']
    scenarios_df['rho_pp'] = scenarios_df['load_pp'] / scenarios_df['cap_pp']

    scenarios_df['sqrt_load_ldr'] = np.sqrt(scenarios_df['load_ldr'])
    scenarios_df['sqrt_load_pp'] = np.sqrt(scenarios_df['load_pp'])

    # Compute queueing approximation terms
    scenarios_df['prob_blockedby_pp_approx'] = scenarios_df.apply(lambda x: obnetwork.prob_blockedby_pp_hat(
        x.arrival_rate, x.mean_los_pp, x.cap_pp, x.pp_cv2_svctime), axis=1)

    scenarios_df['condmeantime_blockedby_pp_approx'] = scenarios_df.apply(
        lambda x: obnetwork.condmeantime_blockedby_pp_hat(
            x.arrival_rate, x.mean_los_pp, x.cap_pp, x.pp_cv2_svctime), axis=1)

    scenarios_df['prob_blockedby_ldr_approx'] = \
        scenarios_df.apply(lambda x: obnetwork.obs_blockedby_ldr_hats(
            x.arrival_rate, x.c_sect_prob, x.mean_los_ldr, x.ldr_cv2_svctime, x.cap_ldr,
            x.mean_los_pp, x.pp_cv2_svctime, x.cap_pp)[2], axis=1)

    scenarios_df['condmeantime_blockedby_ldr_approx'] = \
        scenarios_df.apply(lambda x: obnetwork.obs_blockedby_ldr_hats(
            x.arrival_rate, x.c_sect_prob, x.mean_los_ldr, x.ldr_cv2_svctime, x.cap_ldr,
            x.mean_los_pp, x.pp_cv2_svctime, x.cap_pp)[3], axis=1)

    scenarios_df['ldr_effmean_svctime_approx'] = \
        scenarios_df.apply(lambda x: obnetwork.obs_blockedby_ldr_hats(
            x.arrival_rate, x.c_sect_prob, x.mean_los_ldr, x.ldr_cv2_svctime, x.cap_ldr,
            x.mean_los_pp, x.pp_cv2_svctime, x.cap_pp)[1], axis=1)

    scenarios_df['ldr_eff_load'] = scenarios_df['arrival_rate'] * scenarios_df['ldr_effmean_svctime_approx']
    scenarios_df['ldr_eff_sqrtload'] = np.sqrt(scenarios_df['ldr_eff_load'])

    # Create dataframe consistent with X_ldr_q to be used for predictions
    X_ldr_q_scenarios_df = scenarios_df.loc[:, X_ldr_q_cols]
    X_ldr_noq_scenarios_df = scenarios_df.loc[:, X_ldr_noq_cols]
    X_ldr_occmean_onlyq_df = scenarios_df.loc[:, X_ldr_occmean_onlyq_cols]
    X_ldr_occp95_onlyq_df = scenarios_df.loc[:, X_ldr_occp95_onlyq_cols]
    X_ldr_prob_blockedby_pp_onlyq_df = scenarios_df.loc[:, X_ldr_prob_blockedby_pp_onlyq_cols]
    X_ldr_condmeantime_blockedby_pp_onlyq_df = scenarios_df.loc[:, X_ldr_condmeantime_blockedby_pp_onlyq_cols]
    X_pp_basicq_scenarios_df = scenarios_df.loc[:, X_pp_basicq_cols]
    X_pp_noq_scenarios_df = scenarios_df.loc[:, X_pp_noq_cols]

    meta_inputs_df = scenarios_df.loc[:, meta_input_cols]
    # num_scenarios = len(scenarios_df.index)
    # meta_inputs_df.set_index(np.arange(1, num_scenarios + 1), inplace=True)
    # meta_inputs_df.index.name = 'scenario'

    return {'X_ldr_q': X_ldr_q_scenarios_df,
            'X_ldr_noq': X_ldr_noq_scenarios_df,
            'X_ldr_occmean_onlyq': X_ldr_occmean_onlyq_df,
            'X_ldr_occp95_onlyq': X_ldr_occp95_onlyq_df,
            'X_ldr_prob_blockedby_pp_onlyq': X_ldr_prob_blockedby_pp_onlyq_df,
            'X_ldr_condmeantime_blockedby_pp_onlyq': X_ldr_condmeantime_blockedby_pp_onlyq_df,
            'X_pp_basicq': X_pp_basicq_scenarios_df,
            'X_pp_noq': X_pp_noq_scenarios_df,
            'meta_inputs': meta_inputs_df}


def hyper_erlang_cv2(means, stages, probs):
    mean = np.dot(means, probs)
    rates = [1 / m for m in means]
    m2 = qng.hyper_erlang_moment(rates, stages, probs, 2)
    var = m2 - mean ** 2
    cv2 = var / mean ** 2
    return cv2


if __name__ == '__main__':

    override_args = True

    if override_args:
        mm_experiment_suffix = "exp11"
        perf_curve_scenarios_suffix = "exp11d"
        # Path to scenario yaml file created by scenario_grid.py
        path_scenario_grid_yaml = Path("mm_use", f"scenario_grid_{perf_curve_scenarios_suffix}.yaml")
        path_scenario_csv = Path("mm_use", f"X_performance_curves_{perf_curve_scenarios_suffix}.csv")
        siminout_path = Path("data/siminout")
        matrix_data_path = Path("data")

        with open(path_scenario_grid_yaml, "r") as stream:
            try:
                scenario_grid = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        print(scenario_grid)

        input_scenarios = [scn for scn in itertools.product(*[value for key, value in scenario_grid.items()])]

        cols = ['arrival_rate', 'mean_los_obs', 'mean_los_ldr', 'mean_los_csect',
                'mean_los_pp_noc', 'mean_los_pp_c', 'c_sect_prob',
                'num_erlang_stages_obs', 'num_erlang_stages_ldr', 'num_erlang_stages_csect', 'num_erlang_stages_pp',
                'cap_obs', 'cap_ldr', 'cap_pp']

        input_scenarios_df = pd.DataFrame(input_scenarios, columns=cols)
        num_scenarios = len(input_scenarios_df.index)
        input_scenarios_df.set_index(np.arange(1, num_scenarios + 1), inplace=True)
        input_scenarios_df.index.name = 'scenario'

        scenarios_dfs = make_x_scenarios(input_scenarios_df, mm_experiment_suffix, matrix_data_path)

        # Get each X matrix
        X_ldr_q_scenarios = scenarios_dfs['X_ldr_q']
        X_ldr_noq_scenarios = scenarios_dfs['X_ldr_noq']
        X_ldr_occmean_onlyq_scenarios = scenarios_dfs['X_ldr_occmean_onlyq']
        X_ldr_occp95_onlyq_scenarios = scenarios_dfs['X_ldr_occp95_onlyq']
        X_ldr_prob_blockedby_pp_onlyq_scenarios = scenarios_dfs['X_ldr_prob_blockedby_pp_onlyq']
        X_ldr_condmeantime_blockedby_pp_onlyq_scenarios = scenarios_dfs['X_ldr_condmeantime_blockedby_pp_onlyq']
        X_pp_basicq_scenarios = scenarios_dfs['X_pp_basicq']
        X_pp_noq_scenarios = scenarios_dfs['X_pp_noq']

        # Get results from pickle file
        with open(Path("output", f"ldr_results_{mm_experiment_suffix}.pkl"), 'rb') as pickle_file:
            ldr_results = pickle.load(pickle_file)

        with open(Path("output", f"pp_results_{mm_experiment_suffix}.pkl"), 'rb') as pickle_file:
            pp_results = pickle.load(pickle_file)

        # Get the q based lasso models
        model_ldr_occ_mean_q_lassocv = ldr_results['ldr_occ_mean_q_lassocv_results']['model']
        model_ldr_occ_p95_q_lassocv = ldr_results['ldr_occ_p95_q_lassocv_results']['model']
        model_prob_blockedby_pp_q_lassocv = ldr_results['prob_blockedby_pp_q_lassocv_results']['model']
        model_condmeantime_blockedby_pp_q_lassocv = ldr_results['condmeantime_blockedby_pp_q_lassocv_results']['model']

        # Get the q based lm models
        model_ldr_occ_mean_q_lm = ldr_results['ldr_occ_mean_q_lm_results']['model']
        model_ldr_occ_p95_q_lm = ldr_results['ldr_occ_p95_q_lm_results']['model']
        model_prob_blockedby_pp_q_lm = ldr_results['prob_blockedby_pp_q_lm_results']['model']
        model_condmeantime_blockedby_pp_q_lm = ldr_results['condmeantime_blockedby_pp_q_lm_results']['model']

        # Get the onlyq based lm models
        model_ldr_occ_mean_onlyq_lm = ldr_results['ldr_occ_mean_onlyq_lm_results']['model']
        model_ldr_occ_p95_onlyq_lm = ldr_results['ldr_occ_p95_onlyq_lm_results']['model']
        model_prob_blockedby_pp_onlyq_lm = ldr_results['prob_blockedby_pp_onlyq_lm_results']['model']
        model_condmeantime_blockedby_pp_onlyq_lm = ldr_results['condmeantime_blockedby_pp_onlyq_lm_results']['model']

        # Get the noq based lm models
        model_ldr_occ_mean_noq_lm = ldr_results['ldr_occ_mean_noq_lm_results']['model']
        model_ldr_occ_p95_noq_lm = ldr_results['ldr_occ_p95_noq_lm_results']['model']
        model_prob_blockedby_pp_noq_lm = ldr_results['prob_blockedby_pp_noq_lm_results']['model']
        model_condmeantime_blockedby_pp_noq_lm = ldr_results['condmeantime_blockedby_pp_noq_lm_results']['model']

        # Get the noq based poly models
        model_ldr_occ_mean_noq_poly = ldr_results['ldr_occ_mean_noq_poly_results']['model']
        model_ldr_occ_p95_noq_poly = ldr_results['ldr_occ_p95_noq_poly_results']['model']
        model_prob_blockedby_pp_noq_poly = ldr_results['prob_blockedby_pp_noq_poly_results']['model']
        model_condmeantime_blockedby_pp_noq_poly = ldr_results['condmeantime_blockedby_pp_noq_poly_results']['model']

        # Get the pp basicq based lm models
        model_pp_occ_mean_basicq_lassocv = pp_results['pp_occ_mean_basicq_lassocv_results']['model']
        model_pp_occ_p95_basicq_lassocv = pp_results['pp_occ_p95_basicq_lassocv_results']['model']

        # Get the pp noq based poly models
        model_pp_occ_mean_noq_poly = pp_results['pp_occ_mean_noq_poly_results']['model']
        model_pp_occ_p95_noq_poly = pp_results['pp_occ_p95_noq_poly_results']['model']

        # Make the predictions for each scenario
        scenarios_io_df = X_ldr_q_scenarios.copy()
        scenarios_io_df['pred_ldr_occ_mean_q_lassocv'] = model_ldr_occ_mean_q_lassocv.predict(X_ldr_q_scenarios)
        scenarios_io_df['pred_ldr_occ_p95_q_lassocv'] = model_ldr_occ_p95_q_lassocv.predict(X_ldr_q_scenarios)
        scenarios_io_df['pred_prob_blockedby_pp_q_lassocv'] = \
            model_prob_blockedby_pp_q_lassocv.predict(X_ldr_q_scenarios)
        scenarios_io_df['pred_condmeantime_blockedby_pp_q_lassocv'] = \
            model_condmeantime_blockedby_pp_q_lassocv.predict(X_ldr_q_scenarios)

        scenarios_io_df['pred_ldr_occ_mean_q_lm'] = model_ldr_occ_mean_q_lm.predict(X_ldr_q_scenarios)
        scenarios_io_df['pred_ldr_occ_p95_q_lm'] = model_ldr_occ_p95_q_lm.predict(X_ldr_q_scenarios)
        scenarios_io_df['pred_prob_blockedby_pp_q_lm'] = \
            model_prob_blockedby_pp_q_lm.predict(X_ldr_q_scenarios)
        scenarios_io_df['pred_condmeantime_blockedby_pp_q_lm'] = \
            model_condmeantime_blockedby_pp_q_lm.predict(X_ldr_q_scenarios)

        scenarios_io_df['pred_ldr_occ_mean_onlyq_lm'] = \
            model_ldr_occ_mean_onlyq_lm.predict(X_ldr_occmean_onlyq_scenarios)
        scenarios_io_df['pred_ldr_occ_p95_onlyq_lm'] = \
            model_ldr_occ_p95_onlyq_lm.predict(X_ldr_occp95_onlyq_scenarios)
        scenarios_io_df['pred_prob_blockedby_pp_onlyq_lm'] = \
            model_prob_blockedby_pp_onlyq_lm.predict(X_ldr_prob_blockedby_pp_onlyq_scenarios)
        scenarios_io_df['pred_condmeantime_blockedby_pp_onlyq_lm'] = \
            model_condmeantime_blockedby_pp_onlyq_lm.predict(X_ldr_condmeantime_blockedby_pp_onlyq_scenarios)

        # noq lm
        scenarios_io_df['pred_ldr_occ_mean_noq_lm'] = model_ldr_occ_mean_noq_lm.predict(X_ldr_noq_scenarios)
        scenarios_io_df['pred_ldr_occ_p95_noq_lm'] = model_ldr_occ_p95_noq_lm.predict(X_ldr_noq_scenarios)
        scenarios_io_df['pred_prob_blockedby_pp_noq_lm'] = \
            model_prob_blockedby_pp_noq_lm.predict(X_ldr_noq_scenarios)
        scenarios_io_df['pred_condmeantime_blockedby_pp_noq_lm'] = \
            model_condmeantime_blockedby_pp_noq_lm.predict(X_ldr_noq_scenarios)

        # noq poly
        scenarios_io_df['pred_ldr_occ_mean_noq_poly'] = model_ldr_occ_mean_noq_poly.predict(X_ldr_noq_scenarios)
        scenarios_io_df['pred_ldr_occ_p95_noq_poly'] = model_ldr_occ_p95_noq_poly.predict(X_ldr_noq_scenarios)
        scenarios_io_df['pred_prob_blockedby_pp_noq_poly'] = \
            model_prob_blockedby_pp_noq_poly.predict(X_ldr_noq_scenarios)
        scenarios_io_df['pred_condmeantime_blockedby_pp_noq_poly'] = \
            model_condmeantime_blockedby_pp_noq_poly.predict(X_ldr_noq_scenarios)

        # basicq pp
        scenarios_io_df['pred_pp_occ_mean_basicq_lassocv'] = \
            model_pp_occ_mean_basicq_lassocv.predict(X_pp_basicq_scenarios)
        scenarios_io_df['pred_pp_occ_p95_basicq_lassocv'] = \
            model_pp_occ_p95_basicq_lassocv.predict(X_pp_basicq_scenarios)

        # noq pp
        scenarios_io_df['pred_pp_occ_mean_noq_poly'] = \
            model_pp_occ_mean_noq_poly.predict(X_pp_noq_scenarios)
        scenarios_io_df['pred_pp_occ_p95_noq_poly'] = \
            model_pp_occ_p95_noq_poly.predict(X_pp_noq_scenarios)

        scenarios_io_df.to_csv(path_scenario_csv, index=True)
        print(f'scenarios_io_df written to {path_scenario_csv}')

        # Create meta inputs scenario file
        meta_inputs_df = scenarios_dfs['meta_inputs']
        meta_inputs_path = Path(siminout_path, perf_curve_scenarios_suffix,
                                f'{perf_curve_scenarios_suffix}_obflow06_metainputs_pc.csv')
        meta_inputs_df.to_csv(meta_inputs_path, index=True)
        print(f'meta_inputs_df written to {meta_inputs_path}')

