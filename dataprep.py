import sys
from pathlib import Path
import argparse


import pandas as pd

def create_x_y(experiment, raw_data_path, output_data_path):
    """Read main data file created by simulation output processing and create X and y dataframes.

    :param data_path: str or Path object
    :return:
    """

    xy_df = pd.read_csv(raw_data_path, index_col=0)

    # Define which columns are in which matrices starting with no queueing vars
    X_pp_noq_cols = ['arrival_rate', 'mean_los_pp',
                     'c_sect_prob', 'cap_pp']

    X_ldr_noq_cols = ['arrival_rate', 'mean_los_obs', 'mean_los_ldr', 'cap_ldr',
                      'mean_los_pp', 'c_sect_prob', 'cap_pp']

    X_obs_noq_cols = ['arrival_rate', 'mean_los_obs', 'cap_obs', 'mean_los_ldr', 'cap_ldr',
                      'mean_los_pp', 'c_sect_prob', 'cap_pp']

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
    X_pp_noq = xy_df.loc[:, X_pp_noq_cols]
    X_ldr_noq = xy_df.loc[:, X_ldr_noq_cols]
    X_obs_noq = xy_df.loc[:, X_obs_noq_cols]

    # PP
    X_pp_basicq = xy_df.loc[:, X_pp_basicq_cols]
    X_pp_basicq['sqrt_load_pp'] = X_pp_basicq['load_pp'] ** 0.5

    # LDR
    X_ldr_basicq = xy_df.loc[:, X_ldr_basicq_cols]
    X_ldr_basicq['sqrt_load_ldr'] = X_ldr_basicq['load_ldr'] ** 0.5
    X_ldr_basicq['sqrt_load_pp'] = X_ldr_basicq['load_pp'] ** 0.5

    X_ldr_q = xy_df.loc[:, X_ldr_q_cols]
    X_ldr_q['sqrt_load_ldr'] = X_ldr_q['load_ldr'] ** 0.5
    X_ldr_q['sqrt_load_pp'] = X_ldr_q['load_pp'] ** 0.5

    # OBS
    X_obs_basicq = xy_df.loc[:, X_obs_basicq_cols]
    X_obs_basicq['sqrt_load_obs'] = X_obs_basicq['load_obs'] ** 0.5
    X_obs_basicq['sqrt_load_ldr'] = X_obs_basicq['load_ldr'] ** 0.5
    X_obs_basicq['sqrt_load_pp'] = X_obs_basicq['load_pp'] ** 0.5

    X_obs_q = xy_df.loc[:, X_obs_q_cols]
    X_obs_q['sqrt_load_obs'] = X_obs_q['load_obs'] ** 0.5
    X_obs_q['sqrt_load_ldr'] = X_obs_q['load_ldr'] ** 0.5
    X_obs_q['sqrt_load_pp'] = X_obs_q['load_pp'] ** 0.5

    # y vectors
    y_pp_occ_mean = xy_df.loc[:, 'occ_mean_mean_pp']
    y_pp_occ_p95 = xy_df.loc[:, 'occ_mean_p95_pp']
    y_ldr_occ_mean = xy_df.loc[:, 'occ_mean_mean_ldr']
    y_ldr_occ_p95 = xy_df.loc[:, 'occ_mean_p95_ldr']
    y_obs_occ_mean = xy_df.loc[:, 'occ_mean_mean_obs']
    y_obs_occ_p95 = xy_df.loc[:, 'occ_mean_p95_obs']

    y_mean_pct_blocked_by_pp = xy_df.loc[:, 'prob_blockedby_pp_sim']
    y_mean_pct_blocked_by_ldr = xy_df.loc[:, 'prob_blockedby_ldr_sim']
    y_condmeantime_blockedbyldr = xy_df.loc[:, 'condmeantime_blockedbyldr_sim']
    y_condmeantime_blockedbypp = xy_df.loc[:, 'condmeantime_blockedbypp_sim']

    # Write dataframes to csv
    X_pp_noq.to_csv(Path(output_data_path, f'X_pp_noq_{experiment}.csv'))
    X_pp_basicq.to_csv(Path(output_data_path, f'X_pp_basicq_{experiment}.csv'))

    X_ldr_noq.to_csv(Path(output_data_path, f'X_ldr_noq_{experiment}.csv'))
    X_ldr_basicq.to_csv(Path(output_data_path, f'X_ldr_basicq_{experiment}.csv'))
    X_ldr_q.to_csv(Path(output_data_path, f'X_ldr_q_{experiment}.csv'))

    X_obs_noq.to_csv(Path(output_data_path, f'X_obs_noq_{experiment}.csv'))
    X_obs_basicq.to_csv(Path(output_data_path, f'X_obs_basicq_{experiment}.csv'))
    X_obs_q.to_csv(Path(output_data_path, f'X_obs_q_{experiment}.csv'))

    y_pp_occ_mean.to_csv(Path(output_data_path, f'y_pp_occ_mean_{experiment}.csv'))
    y_pp_occ_p95.to_csv(Path(output_data_path, f'y_pp_occ_p95_{experiment}.csv'))
    y_ldr_occ_mean.to_csv(Path(output_data_path, f'y_ldr_occ_mean_{experiment}.csv'))
    y_ldr_occ_p95.to_csv(Path(output_data_path, f'y_ldr_occ_p95_{experiment}.csv'))
    y_obs_occ_mean.to_csv(Path(output_data_path, f'y_obs_occ_mean_{experiment}.csv'))
    y_obs_occ_p95.to_csv(Path(output_data_path, f'y_obs_occ_p95_{experiment}.csv'))

    y_mean_pct_blocked_by_pp.to_csv(Path(output_data_path, f'y_mean_pct_blocked_by_pp_{experiment}.csv'))
    y_mean_pct_blocked_by_ldr.to_csv(Path(output_data_path, f'y_mean_pct_blocked_by_ldr_{experiment}.csv'))
    y_condmeantime_blockedbyldr.to_csv(Path(output_data_path, f'y_condmeantime_blockedbyldr_{experiment}.csv'))
    y_condmeantime_blockedbypp.to_csv(Path(output_data_path, f'y_condmeantime_blockedbypp_{experiment}.csv'))

def process_command_line():
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='obflow_6',
                                     description='Run inpatient OB simulation')

    # Add arguments
    parser.add_argument(
        "experiment", type=str,
        help="String used in output filenames"
    )

    parser.add_argument(
        "raw_data_path", type=str,
        help="Path to csv file containing scenario inputs, summary stats and qng approximations"
    )

    parser.add_argument(
        "output_data_path", type=str,
        help="Path to directory in which to create X and y data files"
    )

    # do the parsing
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = process_command_line()
    override_args = True

    if override_args:
        experiment = "exp10obflow06"
        raw_data_path = Path("data", "raw", "scenario_inputs_summary_qng.csv")
        output_data_path = Path("data")
    else:
        experiment = args.experiment
        raw_data_path = args.raw_data_path
        output_data_path = args.output_data_path

    create_x_y(experiment, raw_data_path, output_data_path)
