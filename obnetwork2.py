import qng
from scipy import optimize
import pandas as pd

def ldr_prob_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap, pp_cv2_svctime):
    """
    Approximate probability of being blocked in ldr waiting for a pp bed.

    Modeling pp as an M/G/c queue and using erlang C approx
    """
    pp_svcrate = 1.0 / pp_mean_svctime
    prob = qng.mgc_prob_wait_erlangc(arr_rate, pp_svcrate, int(pp_cap))

    return prob

def ldr_meantime_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap, pp_cv2_svctime ):

    """
    Approximate unconditional mean time blocked in ldr waiting for a pp bed.

    Modeling pp as an M/G/c queue and using approximation by Kimura.
    """
    pp_svcrate = 1.0 / pp_mean_svctime
    meantime = qng.mgc_mean_qwait_kimura(arr_rate, pp_svcrate, int(pp_cap), pp_cv2_svctime)

    return meantime

def ldr_vartime_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap, pp_cv2_svctime):
    """
    Approximate unconditional variance of time blocked in ldr waiting for a pp bed.

    Modeling pp as an M/G/c queue and using approximation by Whitt.
    """
    pp_svcrate = 1.0 / pp_mean_svctime
    vartime =  qng.ggm_qwait_whitt_varw(arr_rate, pp_svcrate, int(pp_cap), 1.0, pp_cv2_svctime)

    return vartime


def ldr_condmeantime_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap, pp_cv2_svctime):
    """
    Approximate conditional mean time blocked in ldr waiting for a pp bed.

    Modeling pp as an M/G/c queue
    """
    pp_svcrate = 1.0 / pp_mean_svctime
    prob = qng.mgc_prob_wait_erlangc(arr_rate, pp_svcrate, int(pp_cap))
    meantime = qng.mgc_mean_qwait_kimura(arr_rate, pp_svcrate, int(pp_cap), pp_cv2_svctime)
    condmeantime = meantime / prob

    return condmeantime

def ldr_condpctiletime_blockedby_pp_hat(prob, arr_rate, pp_mean_svctime, pp_cv2_svctime, pp_cap):

    pass

    return -1

def _fixedpt_func_mgc_mean_qwait(x ,arr_rate, effsvctime, cap, cv2):

    svcrate = 1.0 / (effsvctime - x)
    return qng.mgc_mean_qwait_kimura(arr_rate, svcrate, cap, cv2)


def obs_blockedby_ldr_hats(arr_rate, csect_rate, ldr_mean_svctime, ldr_cv2_svctime, ldr_cap,
                         pp_mean_svctime, pp_cv2_svctime, pp_cap):

    # Use MGc approximation
    ldr_meantime_blockedby_pp = ldr_meantime_blockedby_pp_hat(arr_rate, pp_mean_svctime, int(pp_cap), pp_cv2_svctime)
    #ldr_vartime_blockedby_pp = ldr_vartime_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap, pp_cv2_svctime)

    # Start with no reduction by queueing time in obs. Only non c-sections can be blocked
    ldr_effmean_svctime_init = ldr_mean_svctime + (1 - csect_rate) * ldr_meantime_blockedby_pp
    #ldr_effvar_svctime_init = ldr_cv2_svctime * (ldr_mean_svctime ** 2) + ldr_vartime_blockedby_pp

    # The following two variables not used right now as we are assuming effective variance is constant
    # Review this next estimate for eff variance in svc time in LDR
    #ldr_effvar_svctime_init = ldr_cv2_svctime * (ldr_mean_svctime ** 2) + ldr_vartime_blockedby_pp
    #ldr_effcv2_svctime_init = ldr_effvar_svctime_init / (ldr_effmean_svctime_init ** 2)

    # Estimate mean time blocked in obs by solving a fixed point problem
    fixedpoint = optimize.fixed_point(_fixedpt_func_mgc_mean_qwait, [0.0],
                                      args=(arr_rate, ldr_effmean_svctime_init, int(ldr_cap), ldr_cv2_svctime))

    meantime_blockedbyldr_fixedpt = fixedpoint[0] # Since optimize.fixed_point returns an array

    # Now update the estimate of effective svc time in LDR
    # Oops! Following line double adds the PP part
    #ldr_effmean_svctime_final = ldr_effmean_svctime_init + (1 - csect_rate) * ldr_meantime_blockedby_pp - meantime_blockedbyldr_fixedpt
    ldr_effmean_svctime_final = ldr_effmean_svctime_init - meantime_blockedbyldr_fixedpt

    # Finally, compute estimate of prob blocked in obs and conditional meantime blocked
    prob_blockedby_ldr = qng.mgc_prob_wait_erlangc(arr_rate, 1.0 / ldr_effmean_svctime_final, int(ldr_cap))
    condmeantime_blockedbyldr = meantime_blockedbyldr_fixedpt / prob_blockedby_ldr

    return (meantime_blockedbyldr_fixedpt, ldr_effmean_svctime_final,
            prob_blockedby_ldr, condmeantime_blockedbyldr)



if __name__ == '__main__':

    train_df = pd.read_csv('mmdata/train_exp9_tandem05_nodischadj.csv')
    #train_df.set_index('scenario', drop=False, inplace=True)

    results = []
    scenarios = range(1,151)

    for scenario in scenarios:

        arr_rate = train_df.ix[scenario-1, 'lam_obs']
        csect_rate = train_df.ix[scenario-1, 'tot_c_rate']
        ldr_mean_svctime = train_df.ix[scenario-1, 'alos_ldr']
        ldr_cv2_svctime = train_df.ix[scenario-1, 'cv2_ldr']
        #ldr_cv2_svctime = train_df.ix[scenario - 1, 'actual_los_cv2_mean_ldr']
        ldr_cap = train_df.ix[scenario-1, 'cap_ldr']
        pp_mean_svctime = train_df.ix[scenario-1, 'alos_pp']
        pp_cv2_svctime = train_df.ix[scenario-1, 'cv2_pp']
        pp_cap = train_df.ix[scenario-1, 'cap_pp']
        sim_mean_waitq_ldr_mean = train_df.ix[scenario-1, 'mean_waitq_ldr_mean']
        sim_mean_pct_waitq_ldr = train_df.ix[scenario-1, 'mean_pct_waitq_ldr']
        sim_actual_los_mean_mean_ldr = train_df.ix[scenario-1, 'actual_los_mean_mean_ldr']
        sim_mean_pct_blocked_by_pp = train_df.ix[scenario-1, 'mean_pct_blocked_by_pp']
        sim_mean_blocked_by_pp_mean = train_df.ix[scenario-1, 'mean_blocked_by_pp_mean']

        ldr_pct_blockedby_pp = ldr_prob_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap, pp_cv2_svctime)
        ldr_meantime_blockedby_pp = ldr_condmeantime_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap, pp_cv2_svctime)
        (obs_meantime_blockedbyldr, ldr_effmean_svctime, obs_prob_blockedby_ldr, obs_condmeantime_blockedbyldr) = \
            obs_blockedby_ldr_hats(arr_rate, csect_rate, ldr_mean_svctime, ldr_cv2_svctime, ldr_cap,
                                   pp_mean_svctime, pp_cv2_svctime, pp_cap)

        scen_results = {'scenario': scenario,
                        'arr_rate': arr_rate,
                        'prob_blockedby_ldr_approx': obs_prob_blockedby_ldr,
                        'prob_blockedby_ldr_sim': sim_mean_pct_waitq_ldr,

                        'condmeantime_blockedbyldr_approx': obs_condmeantime_blockedbyldr * 24.0,
                        'condmeantime_blockedbyldr_sim': sim_mean_waitq_ldr_mean,
                        'ldr_effmean_svctime_approx': ldr_effmean_svctime * 24.0,
                        'ldr_effmean_svctime_sim': sim_actual_los_mean_mean_ldr,
                        'prob_blockedby_pp_approx': ldr_pct_blockedby_pp,
                        'prob_blockedby_pp_sim': sim_mean_pct_blocked_by_pp,
                        'condmeantime_blockedbypp_approx': ldr_meantime_blockedby_pp * 24.0,
                        'condmeantime_blockedbypp_sim': sim_mean_blocked_by_pp_mean}

        results.append(scen_results)

        # print("scenario {}\n".format(scenario))
        # print(results)


    results_df = pd.DataFrame(results)
    print(results_df)

    #results_df.to_csv("obnetwork_approx_vs_sim.csv")
    results_df.to_csv("obnetwork_approx_vs_sim_testing.csv")



