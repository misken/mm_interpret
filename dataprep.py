from pathlib import Path
import argparse
import math

import pandas as pd

from obnetwork import obnetwork
import qng


def create_x_y(exp, sim_input_output_qnq_path, scenarios, output_path):
    """
    Read main data file created by simulation output processing and create X and y dataframes.
    Parameters
    ----------
    exp
    sim_input_output_qnq_path
    output_path

    Returns
    -------

    """

    xy_df = pd.read_csv(sim_input_output_qnq_path, index_col=0)

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
    X_pp_noq = xy_df.loc[scenarios, X_pp_noq_cols]
    X_ldr_noq = xy_df.loc[scenarios, X_ldr_noq_cols]
    X_obs_noq = xy_df.loc[scenarios, X_obs_noq_cols]

    # PP
    X_pp_basicq = xy_df.loc[scenarios, X_pp_basicq_cols]
    X_pp_basicq['sqrt_load_pp'] = X_pp_basicq['load_pp'] ** 0.5

    # LDR
    X_ldr_basicq = xy_df.loc[scenarios, X_ldr_basicq_cols]
    X_ldr_basicq['sqrt_load_ldr'] = X_ldr_basicq['load_ldr'] ** 0.5
    X_ldr_basicq['sqrt_load_pp'] = X_ldr_basicq['load_pp'] ** 0.5

    X_ldr_q = xy_df.loc[scenarios, X_ldr_q_cols]
    X_ldr_q['sqrt_load_ldr'] = X_ldr_q['load_ldr'] ** 0.5
    X_ldr_q['sqrt_load_pp'] = X_ldr_q['load_pp'] ** 0.5

    # OBS
    X_obs_basicq = xy_df.loc[scenarios, X_obs_basicq_cols]
    X_obs_basicq['sqrt_load_obs'] = X_obs_basicq['load_obs'] ** 0.5
    X_obs_basicq['sqrt_load_ldr'] = X_obs_basicq['load_ldr'] ** 0.5
    X_obs_basicq['sqrt_load_pp'] = X_obs_basicq['load_pp'] ** 0.5

    X_obs_q = xy_df.loc[scenarios, X_obs_q_cols]
    X_obs_q['sqrt_load_obs'] = X_obs_q['load_obs'] ** 0.5
    X_obs_q['sqrt_load_ldr'] = X_obs_q['load_ldr'] ** 0.5
    X_obs_q['sqrt_load_pp'] = X_obs_q['load_pp'] ** 0.5

    # y vectors
    y_pp_occ_mean = xy_df.loc[scenarios, 'occ_mean_mean_pp']
    y_pp_occ_p95 = xy_df.loc[scenarios, 'occ_mean_p95_pp']
    y_ldr_occ_mean = xy_df.loc[scenarios, 'occ_mean_mean_ldr']
    y_ldr_occ_p95 = xy_df.loc[scenarios, 'occ_mean_p95_ldr']
    y_obs_occ_mean = xy_df.loc[scenarios, 'occ_mean_mean_obs']
    y_obs_occ_p95 = xy_df.loc[scenarios, 'occ_mean_p95_obs']

    y_mean_pct_blocked_by_pp = xy_df.loc[scenarios, 'prob_blockedby_pp_sim']
    y_mean_pct_blocked_by_ldr = xy_df.loc[scenarios, 'prob_blockedby_ldr_sim']
    y_condmeantime_blockedbyldr = xy_df.loc[scenarios, 'condmeantime_blockedbyldr_sim']
    y_condmeantime_blockedbypp = xy_df.loc[scenarios, 'condmeantime_blockedbypp_sim']

    # Write dataframes to csv
    X_pp_noq.to_csv(Path(output_path, f'X_pp_noq_{exp}.csv'))
    X_pp_basicq.to_csv(Path(output_path, f'X_pp_basicq_{exp}.csv'))

    X_ldr_noq.to_csv(Path(output_path, f'X_ldr_noq_{exp}.csv'))
    X_ldr_basicq.to_csv(Path(output_path, f'X_ldr_basicq_{exp}.csv'))
    X_ldr_q.to_csv(Path(output_path, f'X_ldr_q_{exp}.csv'))

    X_obs_noq.to_csv(Path(output_path, f'X_obs_noq_{exp}.csv'))
    X_obs_basicq.to_csv(Path(output_path, f'X_obs_basicq_{exp}.csv'))
    X_obs_q.to_csv(Path(output_path, f'X_obs_q_{exp}.csv'))

    y_pp_occ_mean.to_csv(Path(output_path, f'y_pp_occ_mean_{exp}.csv'))
    y_pp_occ_p95.to_csv(Path(output_path, f'y_pp_occ_p95_{exp}.csv'))
    y_ldr_occ_mean.to_csv(Path(output_path, f'y_ldr_occ_mean_{exp}.csv'))
    y_ldr_occ_p95.to_csv(Path(output_path, f'y_ldr_occ_p95_{exp}.csv'))
    y_obs_occ_mean.to_csv(Path(output_path, f'y_obs_occ_mean_{exp}.csv'))
    y_obs_occ_p95.to_csv(Path(output_path, f'y_obs_occ_p95_{exp}.csv'))

    y_mean_pct_blocked_by_pp.to_csv(Path(output_path, f'y_mean_pct_blocked_by_pp_{exp}.csv'))
    y_mean_pct_blocked_by_ldr.to_csv(Path(output_path, f'y_mean_pct_blocked_by_ldr_{exp}.csv'))
    y_condmeantime_blockedbyldr.to_csv(Path(output_path, f'y_condmeantime_blockedbyldr_{exp}.csv'))
    y_condmeantime_blockedbypp.to_csv(Path(output_path, f'y_condmeantime_blockedbypp_{exp}.csv'))


def qng_approx_from_inputs(scenario_inputs_summary):
    results = []

    for row in scenario_inputs_summary.iterrows():
        scenario = row[1]['scenario']
        arr_rate = row[1]['arrival_rate']
        c_sect_prob = row[1]['c_sect_prob']
        ldr_mean_svctime = row[1]['mean_los_ldr']
        ldr_cv2_svctime = 1 / row[1]['num_erlang_stages_ldr']
        ldr_cap = row[1]['cap_ldr']
        pp_mean_svctime = c_sect_prob * row[1]['mean_los_pp_c'] + (1 - c_sect_prob) * row[1]['mean_los_pp_noc']

        rates = [1 / row[1]['mean_los_pp_c'], 1 / row[1]['mean_los_pp_noc']]
        probs = [c_sect_prob, 1 - c_sect_prob]
        stages = [int(row[1]['num_erlang_stages_pp']), int(row[1]['num_erlang_stages_pp'])]
        moments = [qng.hyper_erlang_moment(rates, stages, probs, moment) for moment in [1, 2]]
        variance = moments[1] - moments[0] ** 2
        cv2 = variance / moments[0] ** 2

        pp_cv2_svctime = cv2

        pp_cap = row[1]['cap_pp']

        ldr_pct_blockedby_pp = obnetwork.obnetwork.prob_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap,
                                                                         pp_cv2_svctime)
        ldr_meantime_blockedby_pp = obnetwork.obnetwork.condmeantime_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap,
                                                                                      pp_cv2_svctime)
        (obs_meantime_blockedbyldr, ldr_effmean_svctime, obs_prob_blockedby_ldr, obs_condmeantime_blockedbyldr) = \
            obnetwork.obnetwork.obs_blockedby_ldr_hats(arr_rate, c_sect_prob, ldr_mean_svctime, ldr_cv2_svctime, ldr_cap,
                                                      pp_mean_svctime, pp_cv2_svctime, pp_cap)

        scen_results = {'scenario': scenario,
                        'arr_rate': arr_rate,
                        'prob_blockedby_ldr_approx': obs_prob_blockedby_ldr,
                        'condmeantime_blockedbyldr_approx': obs_condmeantime_blockedbyldr,
                        'ldr_effmean_svctime_approx': ldr_effmean_svctime,
                        'prob_blockedby_pp_approx': ldr_pct_blockedby_pp,
                        'condmeantime_blockedbypp_approx': ldr_meantime_blockedby_pp}

        results.append(scen_results)

    results_df = pd.DataFrame(results)
    return results_df


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
        "siminout_qng_path", type=str,
        help="Path to csv file containing scenario inputs, summary stats and qng approximations"
    )

    parser.add_argument(
        "output_data_path", type=str,
        help="Path to directory in which to create X and y data files"
    )

    parser.add_argument(
        "scenario_start", type=str, default=None,
        help="Start of slice object for use in pandas loc selector"
    )

    parser.add_argument(
        "scenario_end", type=str, default=None,
        help="End of slice object for use in pandas loc selector"
    )


    # do the parsing
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    override_args = True

    if override_args:
        experiment = "exp11"
        siminout_qng_path = Path("data", "raw", "scenario_siminout_qng_exp11.csv")
        output_data_path = Path("data")
        scenarios = slice(1, 135)
    else:
        mm_args = process_command_line()
        experiment = mm_args.experiment
        siminout_qng_path = mm_args.siminout_qng_path
        scenario_start = mm_args.scenario_start
        scenario_end = mm_args.scenario_end
        scenarios = slice(int(scenario_start), int(scenario_end))
        output_data_path = mm_args.output_data_path

    create_x_y(experiment, siminout_qng_path, scenarios, output_data_path)
