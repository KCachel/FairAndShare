import numpy as np
import pandas as pd
import time
from src import *
from baselines import *
from metrics import *

def printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method, fairness_ratio, wall_time, total_positions_seen,
             position_seen_prop, group_0, group_1, group_0_val, group_1_val, group_0_cnt, group_1_cnt, sa_count, ra_count, data_name, delta_val):
    # dictionary of lists

    dict = {'subset': subset,
            'subset_scores': subset_scores,
            'delta_val': delta_val,
            'fairness_goal': fairness_goal,
            'utility_ratio': utility_ratio,
            'method': method,
            'fairness_ratio': fairness_ratio,
            'wall_time': wall_time,
            'total_positions_seen': total_positions_seen,
            'position_seen_prop': position_seen_prop,
            'group_0': group_0,
            'group_1': group_1,
            'group_0_val': group_0_val,
            'group_1_val': group_1_val,
            'group_0_cnt': group_0_cnt,
            'group_1_cnt': group_1_cnt,
            'sa_count': sa_count,
            'ra_count': ra_count,
            'data_name': data_name
            }

    results = pd.DataFrame(dict)
    print(results)
    results.to_csv(output_file, index=False)

def execute(dataset, k, run_cnt, output_file):

    if dataset == "lc":
        group_0_str = "a"
        group_1_str = "b"
        L_items = np.load('lowcorr_items.npy')
        L_scores = np.load('L_scores_zipf_law_10lists.npy')
        num_items, num_lists = np.shape(L_items)
        grp_a_size = int(.2 * num_items)
        items = np.arange(0, num_items)
        group_ids = np.zeros_like(items, dtype=int)
        group_ids[np.where(items < grp_a_size)] = 1
        candidate_db = np.vstack((items, group_ids))

    if dataset == "hc":
        group_0_str = "a"
        group_1_str = "b"
        L_items = np.load('highcorr_items.npy')
        L_scores = np.load('L_scores_zipf_law_10lists.npy')
        num_items, num_lists = np.shape(L_items)
        grp_a_size = int(.2 * num_items)
        items = np.arange(0, num_items)
        group_ids = np.zeros_like(items, dtype=int)
        group_ids[np.where(items < grp_a_size)] = 1
        candidate_db = np.vstack((items, group_ids))
    if dataset == "gauss":
        group_0_str = "a"
        group_1_str = "b"
        L_items = np.load('L_items_gaussian.npy')
        L_scores = np.load('L_scores_gaussian.npy')
        num_items, num_lists = np.shape(L_items)
        grp_a_size = int(.2 * num_items)
        items = np.arange(0, num_items)
        group_ids = np.zeros_like(items, dtype=int)
        group_ids[np.where(items < grp_a_size)] = 1
        candidate_db = np.vstack((items, group_ids))
    if dataset == "bank":
        group_0_str = "married"
        group_1_str = "not-married"
        bank_raw = pd.read_csv('bank-additional-full.csv', sep=';')
        np_candidate_mstatus = np.array(bank_raw["marital"])
        mstatus_int = np.array([0 if val == 'married' else 1 for val in list(np_candidate_mstatus)])

        np_candidate_ids = np.arange(0, 41188, 1, dtype=int)
        campaign = np.array(bank_raw["campaign"])
        campaign = np.column_stack((np_candidate_ids, campaign))
        campaign_sorted = np.flipud(campaign[campaign[:, 1].argsort()])

        pdays = np.array(bank_raw["pdays"])
        pdays = np.column_stack((np_candidate_ids, pdays))
        pdays_sorted = np.flipud(pdays[pdays[:, 1].argsort()])

        previous = np.array(bank_raw["previous"])
        previous = np.column_stack((np_candidate_ids, previous))
        previous_sorted = np.flipud(previous[previous[:, 1].argsort()])

        empvarrate = np.array(bank_raw["emp.var.rate"])
        empvarrate = np.column_stack((np_candidate_ids, empvarrate))
        empvarrate_sorted = np.flipud(empvarrate[empvarrate[:, 1].argsort()])

        conspriceidx = np.array(bank_raw["cons.price.idx"])
        conspriceidx = np.column_stack((np_candidate_ids, conspriceidx))
        conspriceidx_sorted = np.flipud(conspriceidx[conspriceidx[:, 1].argsort()])

        euribor3m = np.array(bank_raw["euribor3m"])
        euribor3m = np.column_stack((np_candidate_ids, euribor3m))
        euribor3m_sorted = np.flipud(euribor3m[euribor3m[:, 1].argsort()])

        nremployed = np.array(bank_raw["nr.employed"])
        nremployed = np.column_stack((np_candidate_ids, nremployed))
        nremployed_sorted = np.flipud(nremployed[nremployed[:, 1].argsort()])

        L_items = np.column_stack((campaign_sorted[:, 0],
                                   pdays_sorted[:, 0],
                                   previous_sorted[:, 0],
                                   empvarrate_sorted[:, 0],
                                   conspriceidx_sorted[:, 0],
                                   euribor3m_sorted[:, 0],
                                   nremployed_sorted[:, 0]
                                   ))
        L_items = np.int_(L_items)
        L_scores = np.column_stack((campaign_sorted[:, 1],
                                    pdays_sorted[:, 1],
                                    previous_sorted[:, 1],
                                    empvarrate_sorted[:, 1],
                                    conspriceidx_sorted[:, 1],
                                    euribor3m_sorted[:, 1],
                                    nremployed_sorted[:, 1]
                                    ))

        candidate_db = np.vstack((np_candidate_ids, mstatus_int))
        num_items, num_lists = np.shape(L_items)

    if dataset == "credit":
        group_0_str = "male"
        group_1_str = "female"
        ccd_raw = pd.read_csv('ccd.csv')
        np_candidate_ids = np.array(ccd_raw["ID"])
        np_candidate_gender = np.array(ccd_raw["isMale"])
        gender_int = np.array(
            [0 if val == 1 else 0 for val in list(np_candidate_gender)])  # gender 0 is male 1 is female

        HasHistoryOfOverduePayments = np.array(ccd_raw["HasHistoryOfOverduePayments"])
        HasHistoryOfOverduePayments = np.column_stack((np_candidate_ids, HasHistoryOfOverduePayments))
        HasHistoryOfOverduePayments_sorted = np.flipud(
            HasHistoryOfOverduePayments[HasHistoryOfOverduePayments[:, 1].argsort()])

        MaxBillAmountOverLast6Months = np.array(ccd_raw["MaxBillAmountOverLast6Months"])
        MaxBillAmountOverLast6Months = np.column_stack((np_candidate_ids, MaxBillAmountOverLast6Months))
        MaxBillAmountOverLast6Months_sorted = np.flipud(
            MaxBillAmountOverLast6Months[MaxBillAmountOverLast6Months[:, 1].argsort()])

        MaxPaymentAmountOverLast6Months = np.array(ccd_raw["MaxPaymentAmountOverLast6Months"])
        MaxPaymentAmountOverLast6Months = np.column_stack((np_candidate_ids, MaxPaymentAmountOverLast6Months))
        MaxPaymentAmountOverLast6Months_sorted = np.flipud(
            MaxPaymentAmountOverLast6Months[MaxPaymentAmountOverLast6Months[:, 1].argsort()])

        MonthsWithZeroBalanceOverLast6Months = np.array(ccd_raw["MonthsWithZeroBalanceOverLast6Months"])
        MonthsWithZeroBalanceOverLast6Months = np.column_stack((np_candidate_ids, MonthsWithZeroBalanceOverLast6Months))
        MonthsWithZeroBalanceOverLast6Months_sorted = np.flipud(
            MonthsWithZeroBalanceOverLast6Months[MonthsWithZeroBalanceOverLast6Months[:, 1].argsort()])

        MonthsWithLowSpendingOverLast6Months = np.array(ccd_raw["MonthsWithLowSpendingOverLast6Months"])
        MonthsWithLowSpendingOverLast6Months = np.column_stack((np_candidate_ids, MonthsWithLowSpendingOverLast6Months))
        MonthsWithLowSpendingOverLast6Months_sorted = np.flipud(
            MonthsWithLowSpendingOverLast6Months[MonthsWithLowSpendingOverLast6Months[:, 1].argsort()])

        MonthsWithHighSpendingOverLast6Months = np.array(ccd_raw["MonthsWithHighSpendingOverLast6Months"])
        MonthsWithHighSpendingOverLast6Months = np.column_stack(
            (np_candidate_ids, MonthsWithHighSpendingOverLast6Months))
        MonthsWithHighSpendingOverLast6Months_sorted = np.flipud(
            MonthsWithHighSpendingOverLast6Months[MonthsWithHighSpendingOverLast6Months[:, 1].argsort()])

        MostRecentBillAmount = np.array(ccd_raw["MostRecentBillAmount"])
        MostRecentBillAmount = np.column_stack((np_candidate_ids, MostRecentBillAmount))
        MostRecentBillAmount_sorted = np.flipud(MostRecentBillAmount[MostRecentBillAmount[:, 1].argsort()])

        MostRecentPaymentAmount = np.array(ccd_raw["MostRecentPaymentAmount"])
        MostRecentPaymentAmount = np.column_stack((np_candidate_ids, MostRecentPaymentAmount))
        MostRecentPaymentAmount_sorted = np.flipud(MostRecentPaymentAmount[MostRecentPaymentAmount[:, 1].argsort()])

        TotalOverdueCounts = np.array(ccd_raw["TotalOverdueCounts"])
        TotalOverdueCounts = np.column_stack((np_candidate_ids, TotalOverdueCounts))
        TotalOverdueCounts_sorted = np.flipud(TotalOverdueCounts[TotalOverdueCounts[:, 1].argsort()])

        TotalMonthsOverdue = np.array(ccd_raw["TotalMonthsOverdue"])
        TotalMonthsOverdue = np.column_stack((np_candidate_ids, TotalMonthsOverdue))
        TotalMonthsOverdue_sorted = np.flipud(TotalMonthsOverdue[TotalMonthsOverdue[:, 1].argsort()])

        L_items = np.column_stack((HasHistoryOfOverduePayments_sorted[:, 0],
                                   MaxBillAmountOverLast6Months_sorted[:, 0],
                                   MaxPaymentAmountOverLast6Months_sorted[:, 0],
                                   MonthsWithZeroBalanceOverLast6Months_sorted[:, 0],
                                   MonthsWithLowSpendingOverLast6Months_sorted[:, 0],
                                   MonthsWithHighSpendingOverLast6Months_sorted[:, 0],
                                   MostRecentBillAmount_sorted[:, 0],
                                   MostRecentPaymentAmount_sorted[:, 0],
                                   TotalOverdueCounts_sorted[:, 0],
                                   TotalMonthsOverdue_sorted[:, 0]
                                   ))
        L_scores = np.column_stack((HasHistoryOfOverduePayments_sorted[:, 1],
                                    MaxBillAmountOverLast6Months_sorted[:, 1],
                                    MaxPaymentAmountOverLast6Months_sorted[:, 1],
                                    MonthsWithZeroBalanceOverLast6Months_sorted[:, 1],
                                    MonthsWithLowSpendingOverLast6Months_sorted[:, 1],
                                    MonthsWithHighSpendingOverLast6Months_sorted[:, 1],
                                    MostRecentBillAmount_sorted[:, 1],
                                    MostRecentPaymentAmount_sorted[:, 1],
                                    TotalOverdueCounts_sorted[:, 1],
                                    TotalMonthsOverdue_sorted[:, 1]))

        candidate_db = np.vstack((np_candidate_ids, np_candidate_gender))
        num_items, num_lists = np.shape(L_items)

    #initialize data collectors
    subset = []
    subset_scores = []
    fairness_goal = []
    utility_ratio = []
    method = []
    fairness_ratio = []
    wall_time = []
    total_positions_seen = []
    position_seen_prop = []
    group_0 = []
    group_1 = []
    group_0_val = []
    group_1_val = []
    group_0_cnt = []
    group_1_cnt = []
    sa_count = []
    ra_count = []
    data_name = []
    delta_val = []

    fairness_string = "proportional"
    delta_dict = {}
    delta_dict[0] = "fair-"
    delta_dict[1] = ""
    delta_dict[.1] = "fair-"
    delta_dict[.05] = "fair-"
    np.random.seed(0)




    # #Fagins
    for delta in [1, .1, .05, 0]:
        times = []
        for t in range(0, run_cnt):
            start_time = time.time()
            K_items_FF, K_scores_FF = fairFA(fairness_string, delta, L_items, L_scores, candidate_db, k)
            end_time = time.time()
            times.append(end_time - start_time)
        if delta == 1: MAX_UTIL = np.sum(K_scores_FF)
        set_groups = np.asarray([candidate_db[1, item] for item in K_items_FF])
        selectRt, sp_val = parity(candidate_db[0, :], candidate_db[1, :], K_items_FF, set_groups)
        subset.append(K_items_FF)
        subset_scores.append(K_scores_FF)
        fairness_goal.append(fairness_string)
        utility_ratio.append(np.sum(K_scores_FF)/MAX_UTIL)
        method.append(delta_dict[delta]+'fagins')
        fairness_ratio.append(sp_val)
        wall_time.append(np.mean(times))
        data_name.append(dataset)
        group_0.append(group_0_str)
        group_1.append(group_1_str)
        group_0_val.append(selectRt[0])
        group_1_val.append(selectRt[1])
        _, grp_cnt = np.unique(set_groups, return_counts=True)
        group_0_cnt.append(grp_cnt[0])
        group_1_cnt.append(grp_cnt[1])
        _, _, sa, ra, total_seen = fairFA_perfcounts(fairness_string, delta, L_items, L_scores, candidate_db, k)
        total_positions_seen.append(total_seen)
        position_seen_prop.append(total_seen / (num_items * num_lists))
        sa_count.append(sa)
        ra_count.append(ra)
        delta_val.append(delta)
    printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method, fairness_ratio, wall_time,
             total_positions_seen,
             position_seen_prop, group_0, group_1, group_0_val, group_1_val, group_0_cnt, group_1_cnt, sa_count,
             ra_count, data_name, delta_val)

    # #Thresholds
    # for t_style in ['TA','BPA', 'BPA2']:
    #     for delta in [.1, .05, 0]:
    #         times = []
    #         for t in range(0, run_cnt):
    #             start_time = time.time()
    #             K_items, K_scores = ThresholdFMCS(fairness_string, delta, L_items, L_scores, candidate_db, k, t_style)
    #             end_time = time.time()
    #             times.append(end_time - start_time)
    #         set_groups = np.asarray([candidate_db[1, item] for item in K_items])
    #         selectRt, sp_val = parity(candidate_db[0, :], candidate_db[1, :], K_items, set_groups)
    #         subset.append(K_items)
    #         subset_scores.append(K_scores)
    #         fairness_goal.append(fairness_string)
    #         utility_ratio.append(np.sum(K_scores) / MAX_UTIL)
    #         method.append(delta_dict[delta] + t_style.lower())
    #         fairness_ratio.append(sp_val)
    #         wall_time.append(np.mean(times))
    #         data_name.append(dataset)
    #         group_0.append(group_0_str)
    #         group_1.append(group_1_str)
    #         group_0_val.append(selectRt[0])
    #         group_1_val.append(selectRt[1])
    #         _, grp_cnt = np.unique(set_groups, return_counts=True)
    #         group_0_cnt.append(grp_cnt[0])
    #         group_1_cnt.append(grp_cnt[1])
    #         _, _, sa, ra, total_seen = ThresholdFMCS_perfcounts(fairness_string, delta, L_items, L_scores, candidate_db, k, t_style)
    #         total_positions_seen.append(total_seen)
    #         position_seen_prop.append(total_seen / (num_items * num_lists))
    #         sa_count.append(sa)
    #         ra_count.append(ra)
    #         delta_val.append(delta)
    #     printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method, fairness_ratio, wall_time,
    #              total_positions_seen,
    #              position_seen_prop, group_0, group_1, group_0_val, group_1_val, group_0_cnt, group_1_cnt, sa_count,
    #              ra_count, data_name, delta_val)
    #
    # # GBG Fagins
    # for delta in [.1, .05, 0]:
    #     times = []
    #     for t in range(0, run_cnt):
    #         start_time = time.time()
    #         K_items_FF, K_scores_FF = GBG_fagin(fairness_string, delta, L_items, L_scores, candidate_db, k)
    #         end_time = time.time()
    #         times.append(end_time - start_time)
    #     set_groups = np.asarray([candidate_db[1, item] for item in K_items_FF])
    #     selectRt, sp_val = parity(candidate_db[0, :], candidate_db[1, :], K_items_FF, set_groups)
    #     subset.append(K_items_FF)
    #     subset_scores.append(K_scores_FF)
    #     fairness_goal.append(fairness_string)
    #     utility_ratio.append(np.sum(K_scores_FF) / MAX_UTIL)
    #     method.append(delta_dict[delta] + "GBG" + 'fagins')
    #     fairness_ratio.append(sp_val)
    #     wall_time.append(np.mean(times))
    #     data_name.append(dataset)
    #     group_0.append(group_0_str)
    #     group_1.append(group_1_str)
    #     group_0_val.append(selectRt[0])
    #     group_1_val.append(selectRt[1])
    #     _, grp_cnt = np.unique(set_groups, return_counts=True)
    #     group_0_cnt.append(np.count_nonzero(set_groups==0))
    #     group_1_cnt.append(np.count_nonzero(set_groups))
    #     _, _, sa, ra, total_seen = GBG_fagin_perfcounts(fairness_string, delta, L_items, L_scores, candidate_db, k)
    #     total_positions_seen.append(total_seen)
    #     position_seen_prop.append(total_seen / (num_items * num_lists))
    #     sa_count.append(sa)
    #     ra_count.append(ra)
    #     delta_val.append(delta)
    # printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method, fairness_ratio, wall_time,
    #          total_positions_seen,
    #          position_seen_prop, group_0, group_1, group_0_val, group_1_val, group_0_cnt, group_1_cnt, sa_count,
    #          ra_count, data_name, delta_val)

    # GBG Thresholds
    for t_style in ['BPA2']:
        for delta in [.1, .05, 0]:
            times = []
            for t in range(0, run_cnt):
                start_time = time.time()
                K_items, K_scores = GBG_threshold(fairness_string, delta, L_items, L_scores, candidate_db, k,
                                                  t_style)
                end_time = time.time()
                times.append(end_time - start_time)
            set_groups = np.asarray([candidate_db[1, item] for item in K_items])
            selectRt, sp_val = parity(candidate_db[0, :], candidate_db[1, :], K_items, set_groups)
            subset.append(K_items)
            subset_scores.append(K_scores)
            fairness_goal.append(fairness_string)
            utility_ratio.append(np.sum(K_scores) / MAX_UTIL)
            method.append(delta_dict[delta] + 'GBG' + t_style.lower())
            fairness_ratio.append(sp_val)
            wall_time.append(np.mean(times))
            data_name.append(dataset)
            group_0.append(group_0_str)
            group_1.append(group_1_str)
            group_0_val.append(selectRt[0])
            group_1_val.append(selectRt[1])
            _, grp_cnt = np.unique(set_groups, return_counts=True)
            group_0_cnt.append(np.count_nonzero(set_groups == 0))
            group_1_cnt.append(np.count_nonzero(set_groups))
            _, _, sa, ra, total_seen = GBG_threshold_perfcounts(fairness_string, delta, L_items, L_scores,
                                                                candidate_db, k, t_style)
            total_positions_seen.append(total_seen)
            position_seen_prop.append(total_seen / (num_items * num_lists))
            sa_count.append(sa)
            ra_count.append(ra)
            delta_val.append(delta)
        printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method, fairness_ratio,
                 wall_time,
                 total_positions_seen,
                 position_seen_prop, group_0, group_1, group_0_val, group_1_val, group_0_cnt, group_1_cnt, sa_count,
                 ra_count, data_name, delta_val)

    # # Greedy Fair
    # for delta in [.1, .05, 0]:
    #     times = []
    #     for t in range(0, run_cnt):
    #         start_time = time.time()
    #         K_items_FF, K_scores_FF = greedyFMC(fairness_string, delta, L_items, L_scores, candidate_db, k)
    #         end_time = time.time()
    #         times.append(end_time - start_time)
    #     set_groups = np.asarray([candidate_db[1, item] for item in K_items_FF])
    #     selectRt, sp_val = parity(candidate_db[0, :], candidate_db[1, :], K_items_FF, set_groups)
    #     subset.append(K_items_FF)
    #     subset_scores.append(K_scores_FF)
    #     fairness_goal.append(fairness_string)
    #     utility_ratio.append(np.sum(K_scores_FF) / MAX_UTIL)
    #     method.append(delta_dict[delta] + 'Greedy_FMC')
    #     fairness_ratio.append(sp_val)
    #     wall_time.append(np.mean(times))
    #     data_name.append(dataset)
    #     group_0.append(group_0_str)
    #     group_1.append(group_1_str)
    #     group_0_val.append(selectRt[0])
    #     group_1_val.append(selectRt[1])
    #     _, grp_cnt = np.unique(set_groups, return_counts=True)
    #     group_0_cnt.append(np.count_nonzero(set_groups == 0))
    #     group_1_cnt.append(np.count_nonzero(set_groups))
    #     _, _, sa, ra, total_seen = greedyFMC_perfcounts(fairness_string, delta, L_items, L_scores, candidate_db, k)
    #     total_positions_seen.append(total_seen)
    #     position_seen_prop.append(total_seen / (num_items * num_lists))
    #     sa_count.append(sa)
    #     ra_count.append(ra)
    #     delta_val.append(delta)
    # printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method, fairness_ratio, wall_time,
    #          total_positions_seen,
    #          position_seen_prop, group_0, group_1, group_0_val, group_1_val, group_0_cnt, group_1_cnt, sa_count,
    #          ra_count, data_name, delta_val)
    #
    # # DIVTOPK
    # for delta in [1, .1, .05, 0]:
    #     times = []
    #     for t in range(0, run_cnt):
    #         start_time = time.time()
    #         K_items_FF, K_scores_FF = baseline_divtopk(fairness_string, delta, L_items, L_scores, candidate_db, k)
    #         end_time = time.time()
    #         times.append(end_time - start_time)
    #     set_groups = np.asarray([candidate_db[1, item] for item in K_items_FF])
    #     selectRt, sp_val = parity(candidate_db[0, :], candidate_db[1, :], K_items_FF, set_groups)
    #     subset.append(K_items_FF)
    #     subset_scores.append(K_scores_FF)
    #     fairness_goal.append(fairness_string)
    #     utility_ratio.append(np.sum(K_scores_FF) / MAX_UTIL)
    #     method.append('divtopk')
    #     fairness_ratio.append(sp_val)
    #     wall_time.append(np.mean(times))
    #     data_name.append(dataset)
    #     group_0.append(group_0_str)
    #     group_1.append(group_1_str)
    #     group_0_val.append(selectRt[0])
    #     group_1_val.append(selectRt[1])
    #     _, grp_cnt = np.unique(set_groups, return_counts=True)
    #     group_0_cnt.append(grp_cnt[0])
    #     group_1_cnt.append(grp_cnt[1])
    #     #_, _, sa, ra, total_seen = baseline_divtopk_perfcounts(fairness_string, delta, L_items, L_scores, candidate_db, k)
    #     total_positions_seen.append(total_seen)
    #     position_seen_prop.append(total_seen / (num_items * num_lists))
    #     sa_count.append(num_items)
    #     ra_count.append(0)
    #     delta_val.append(delta)
    # printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method, fairness_ratio, wall_time,
    #          total_positions_seen,
    #          position_seen_prop, group_0, group_1, group_0_val, group_1_val, group_0_cnt, group_1_cnt, sa_count,
    #          ra_count, data_name, delta_val)

iter = 1
# data = 'lc'
# execute(data, 100, iter, 'GBGproportional_study_delta'+ data +'.csv')
# data ='hc'
# execute(data, 100, iter, 'GBGproportional_study_delta'+ data +'.csv')
data = 'gauss'
execute(data, 100, iter, 'GBGproportional_study_delta'+ data +'.csv')
# data = 'bank'
# execute(data, 80, iter, 'GBGproportional_study_delta'+ data +'.csv')
# data = 'credit'
# execute(data, 150, iter, 'GBGproportional_study_delta'+ data +'.csv')

