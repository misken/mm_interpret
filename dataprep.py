import sys
from pathlib import Path
import pandas as pd

def create_x_y(raw_data_path, output_data_path):
    """Read main data file created by simulation output processing and create X and y dataframes.

    :param data_path: str or Path object
    :return:
    """

    obsim_mm_means_df = pd.read_csv(raw_data_path, index_col=0)

    # Define which columns are in which matrices starting with no queueing vars
    X_pp_noq_cols = ['lam_obs', 'alos_pp', 'alos_pp_noc', 'alos_pp_c',
                     'tot_c_rate', 'cap_pp']

    X_ldr_noq_cols = ['lam_obs', 'alos_obs', 'alos_ldr', 'cap_ldr',
                      'alos_pp', 'alos_pp_noc', 'alos_pp_c',
                      'tot_c_rate', 'cap_pp']

    X_obs_noq_cols = ['lam_obs', 'alos_obs', 'cap_obs', 'alos_ldr', 'cap_ldr',
                      'alos_pp', 'alos_pp_noc', 'alos_pp_c',
                      'tot_c_rate', 'cap_pp']

    # For "basicq" matrices, only load and rho variables are added
    X_pp_basicq_cols = X_pp_noq_cols.copy()
    X_pp_basicq_cols.extend(['load_pp', 'rho_pp'])

    X_ldr_basicq_cols = X_ldr_noq_cols.copy()
    X_ldr_basicq_cols.extend(['load_ldr', 'rho_ldr', 'load_pp', 'rho_pp'])

    X_obs_basicq_cols = X_obs_noq_cols.copy()
    X_obs_basicq_cols.extend(['load_obs', 'rho_obs', 'load_ldr',
                              'rho_ldr', 'load_pp', 'rho_pp'])

    # For "q" matrices, include additional queueing approximations (not applicable
    # to PP since unaffected by upstream unit and has no downstream unit

    # LDR can have LOS shortened by patients blocked in OBS and have LOS lengthened
    # by patients blocked in LDR by PP
    X_ldr_q_cols = X_ldr_basicq_cols.copy()
    X_ldr_q_cols.extend(['prob_blockedby_pp_approx', 'condmeantime_blockedbypp_approx',
                         'prob_blockedby_ldr_approx', 'condmeantime_blockedbyldr_approx',
                         'ldr_effmean_svctime_approx'])

    # OBS modeled as infinite capacity system but time in system impacted by
    # congestion in the downstream units.
    X_obs_q_cols = X_obs_basicq_cols.copy()
    X_obs_q_cols.extend(['prob_blockedby_pp_approx', 'condmeantime_blockedbypp_approx',
                         'prob_blockedby_ldr_approx', 'condmeantime_blockedbyldr_approx',
                         'ldr_effmean_svctime_approx'])


    # Create dataframes based on the column specs above
    X_pp_noq = obsim_mm_means_df.loc[:, X_pp_noq_cols]
    X_ldr_noq = obsim_mm_means_df.loc[:, X_ldr_noq_cols]
    X_obs_noq = obsim_mm_means_df.loc[:, X_obs_noq_cols]

    # PP
    X_pp_basicq = obsim_mm_means_df.loc[:, X_pp_basicq_cols]
    X_pp_basicq['sqrt_load_pp'] = X_pp_basicq['load_pp'] ** 0.5

    # LDR
    X_ldr_basicq = obsim_mm_means_df.loc[:, X_ldr_basicq_cols]
    X_ldr_basicq['sqrt_load_ldr'] = X_ldr_basicq['load_ldr'] ** 0.5
    X_ldr_basicq['sqrt_load_pp'] = X_ldr_basicq['load_pp'] ** 0.5

    X_ldr_q = obsim_mm_means_df.loc[:, X_ldr_q_cols]
    X_ldr_q['sqrt_load_ldr'] = X_ldr_q['load_ldr'] ** 0.5
    X_ldr_q['sqrt_load_pp'] = X_ldr_q['load_pp'] ** 0.5

    # OBS
    X_obs_basicq = obsim_mm_means_df.loc[:, X_obs_basicq_cols]
    X_obs_basicq['sqrt_load_obs'] = X_obs_basicq['load_obs'] ** 0.5
    X_obs_basicq['sqrt_load_ldr'] = X_obs_basicq['load_ldr'] ** 0.5
    X_obs_basicq['sqrt_load_pp'] = X_obs_basicq['load_pp'] ** 0.5

    X_obs_q = obsim_mm_means_df.loc[:, X_obs_q_cols]
    X_obs_q['sqrt_load_obs'] = X_obs_q['load_obs'] ** 0.5
    X_obs_q['sqrt_load_ldr'] = X_obs_q['load_ldr'] ** 0.5
    X_obs_q['sqrt_load_pp'] = X_obs_q['load_pp'] ** 0.5

    # y vectors
    y_pp_occ_mean = obsim_mm_means_df.loc[:, 'occ_mean_mean_pp']
    y_pp_occ_p95 = obsim_mm_means_df.loc[:, 'occ_mean_p95_pp']
    y_ldr_occ_mean = obsim_mm_means_df.loc[:, 'occ_mean_mean_ldr']
    y_ldr_occ_p95 = obsim_mm_means_df.loc[:, 'occ_mean_p95_ldr']
    y_obs_occ_mean = obsim_mm_means_df.loc[:, 'occ_mean_mean_obs']
    y_obs_occ_p95 = obsim_mm_means_df.loc[:, 'occ_mean_p95_obs']

    y_mean_pct_blocked_by_pp = obsim_mm_means_df.loc[:, 'prob_blockedby_pp_sim']
    y_mean_pct_blocked_by_ldr = obsim_mm_means_df.loc[:, 'prob_blockedby_ldr_sim']
    y_condmeantime_blockedbyldr = obsim_mm_means_df.loc[:, 'condmeantime_blockedbyldr_sim']
    y_condmeantime_blockedbypp = obsim_mm_means_df.loc[:, 'condmeantime_blockedbypp_sim']

    # Write dataframes to csv
    X_pp_noq.to_csv(Path(output_data_path, 'X_pp_noq.csv'))
    X_pp_basicq.to_csv(Path(output_data_path, 'X_pp_basicq.csv'))

    X_ldr_noq.to_csv(Path(output_data_path, 'X_ldr_noq.csv'))
    X_ldr_basicq.to_csv(Path(output_data_path, 'X_ldr_basicq.csv'))
    X_ldr_q.to_csv(Path(output_data_path, 'X_ldr_q.csv'))

    X_obs_noq.to_csv(Path(output_data_path, 'X_obs_noq.csv'))
    X_obs_basicq.to_csv(Path(output_data_path, 'X_obs_basicq.csv'))
    X_obs_q.to_csv(Path(output_data_path, 'X_obs_q.csv'))

    y_pp_occ_mean.to_csv(Path(output_data_path, 'y_pp_occ_mean.csv'))
    y_pp_occ_p95.to_csv(Path(output_data_path, 'y_pp_occ_p95.csv'))
    y_ldr_occ_mean.to_csv(Path(output_data_path, 'y_ldr_occ_mean.csv'))
    y_ldr_occ_p95.to_csv(Path(output_data_path, 'y_ldr_occ_p95.csv'))
    y_obs_occ_mean.to_csv(Path(output_data_path, 'y_obs_occ_mean.csv'))
    y_obs_occ_p95.to_csv(Path(output_data_path, 'y_obs_occ_p95.csv'))

    y_mean_pct_blocked_by_pp.to_csv(Path(output_data_path, 'y_mean_pct_blocked_by_pp.csv'))
    y_mean_pct_blocked_by_ldr.to_csv(Path(output_data_path, 'y_mean_pct_blocked_by_ldr.csv'))
    y_condmeantime_blockedbyldr.to_csv(Path(output_data_path, 'y_condmeantime_blockedbyldr.csv'))
    y_condmeantime_blockedbypp.to_csv(Path(output_data_path, 'y_condmeantime_blockedbypp.csv'))

if __name__ == '__main__':
    raw_data_path = Path("data", "raw", "obsim_mm_means_df.csv")
    output_data_path = Path("data")
    create_x_y(raw_data_path, output_data_path)
