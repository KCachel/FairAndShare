import numpy as np
import pandas as pd
import time
from src import *
from baselines import *
from metrics import *

def count_grp_members(pool_groups,subset_groups):
    unique_grps = np.unique(pool_groups)
    num_items = []
    for grp in unique_grps:
        subset_mask = subset_groups == grp
        num_items.append(np.count_nonzero(subset_mask))
    return np.asarray(num_items)


def printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method, wall_time,
             total_positions_seen, position_seen_prop, sa_count, ra_count, data_name, count_White, count_Black, count_Asian,
             count_AmIn, count_Other, r_value):
    # dictionary of lists

    dict = {'subset': subset,
            'r_value': r_value,
            'subset_scores': subset_scores,
            'fairness_goal': fairness_goal,
            'utility_ratio': utility_ratio,
            'method': method,
            'wall_time': wall_time,
            'total_positions_seen': total_positions_seen,
            'position_seen_prop': position_seen_prop,
            'count_White': count_White,
            'count_Black': count_Black,
            'count_Asian': count_Asian,
            'count_AmIn': count_AmIn,
            'count_Other': count_Other,
            'sa_count': sa_count,
            'ra_count': ra_count,
            'data_name': data_name
            }

    results = pd.DataFrame(dict)
    print(results)
    results.to_csv(output_file, index=False)

def execute(dataset, k, run_cnt, output_file):

    if dataset == "adult":
        adult_raw = pd.read_csv('adult.csv', sep=',')
        np_candidate_race = np.array(adult_raw["race"])

        race_int = []
        for val in list(np_candidate_race):
            if val == 'White':  # W
                race_int.append(0)
            if val == 'Black':  # B
                race_int.append(1)
            if val == 'Asian-Pac-Islander':  # A
                race_int.append(2)
            if val == 'Amer-Indian-Eskimo':  # I
                race_int.append(3)
            if val == 'Other':  # O
                race_int.append(4)
        race = np.asarray(race_int)
        np_candidate_ids = np.arange(0, 48842, 1, dtype=int)

        educationnum = np.array(adult_raw["education-num"])
        educationnum = np.column_stack((np_candidate_ids, educationnum))
        educationnum_sorted = np.flipud(educationnum[educationnum[:, 1].argsort()])

        capitalgain = np.array(adult_raw["capital-gain"])
        capitalgain = np.column_stack((np_candidate_ids, capitalgain))
        capitalgain_sorted = np.flipud(capitalgain[capitalgain[:, 1].argsort()])

        capitalloss = np.array(adult_raw["capital-loss"])
        capitalloss = np.column_stack((np_candidate_ids, capitalloss))
        capitalloss_sorted = np.flipud(capitalloss[capitalloss[:, 1].argsort()])

        hoursperweek = np.array(adult_raw["hours-per-week"])
        hoursperweek = np.column_stack((np_candidate_ids, hoursperweek))
        hoursperweek_sorted = np.flipud(hoursperweek[hoursperweek[:, 1].argsort()])

        L_items = np.column_stack((educationnum_sorted[:, 0],
                                   capitalgain_sorted[:, 0],
                                   capitalloss_sorted[:, 0],
                                   hoursperweek_sorted[:, 0]
                                   ))
        L_items = np.int_(L_items)
        L_scores = np.column_stack((educationnum_sorted[:, 1],
                                    capitalgain_sorted[:, 1],
                                    capitalloss_sorted[:, 1],
                                    hoursperweek_sorted[:, 1]
                                    ))

        candidate_db = np.vstack((np_candidate_ids, race))
        num_items, num_lists = np.shape(L_items)

    #initialize data collectors
    subset = []
    subset_scores = []
    fairness_goal = []
    utility_ratio = []
    method = []
    wall_time = []
    total_positions_seen = []
    position_seen_prop = []
    count_White = []
    count_Black = []
    count_Asian = []
    count_AmIn = []
    count_Other = []
    sa_count = []
    ra_count = []
    data_name = []
    r_value = []

    delta_dict = {}
    delta_dict[0] = "fair-"
    delta_dict[1] = ""

    # Fagins
    for r_cnt in [0,1,2,3,4,5,6,7,8,9,10]:
        fairness_string = 'rooney ' + str(r_cnt)
        r = r_cnt
        delta = 0
        times = []
        for t in range(0, run_cnt):
            start_time = time.time()
            K_items_FF, K_scores_FF = fairFA(fairness_string, delta, L_items, L_scores, candidate_db, k)
            end_time = time.time()
            times.append(end_time - start_time)

        set_groups = np.asarray([candidate_db[1, item] for item in K_items_FF])
        if r_cnt == 0:
            max_util = np.sum(K_scores_FF)
        subset.append(K_items_FF)
        subset_scores.append(K_scores_FF)
        fairness_goal.append(fairness_string)
        utility_ratio.append(np.sum(K_scores_FF)/max_util)
        method.append(delta_dict[delta]+'fagins')
        wall_time.append(np.mean(times))
        data_name.append(dataset)
        grp_cnt = count_grp_members(candidate_db[1, :], set_groups)
        count_White.append(grp_cnt[0])
        count_Black.append(grp_cnt[1])
        count_Asian.append(grp_cnt[2])
        count_AmIn.append(grp_cnt[3])
        count_Other.append(grp_cnt[4])
        _, _, sa, ra, total_seen = fairFA_perfcounts(fairness_string, delta, L_items, L_scores, candidate_db, k)
        total_positions_seen.append(total_seen)
        position_seen_prop.append(total_seen / (num_items * num_lists))
        sa_count.append(sa)
        ra_count.append(ra)
        r_value.append(r)
    printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method, wall_time,
             total_positions_seen,
             position_seen_prop, sa_count,
             ra_count,
             data_name, count_White, count_Black, count_Asian, count_AmIn, count_Other, r_value)
    # #Thresholds
    for t_style in ['TA','BPA', 'BPA2']:
        for r_cnt in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            fairness_string = 'rooney ' + str(r_cnt)
            r = r_cnt
            delta = 0
            times = []
            for t in range(0, run_cnt):
                start_time = time.time()
                K_items, K_scores = ThresholdFMCS(fairness_string, delta, L_items, L_scores, candidate_db, k, t_style)
                end_time = time.time()
                times.append(end_time - start_time)
            if r_cnt == 0: max_util = np.sum(K_scores)
            set_groups = np.asarray([candidate_db[1, item] for item in K_items])
            subset.append(K_items)
            subset_scores.append(K_scores)
            fairness_goal.append(fairness_string)
            utility_ratio.append(np.sum(K_scores) / max_util)
            method.append(delta_dict[delta] + t_style.lower())
            wall_time.append(np.mean(times))
            data_name.append(dataset)
            grp_cnt = count_grp_members(candidate_db[1, :], set_groups)
            count_White.append(grp_cnt[0])
            count_Black.append(grp_cnt[1])
            count_Asian.append(grp_cnt[2])
            count_AmIn.append(grp_cnt[3])
            count_Other.append(grp_cnt[4])
            _, _, sa, ra, total_seen = ThresholdFMCS_perfcounts(fairness_string, delta, L_items, L_scores, candidate_db, k, t_style)
            total_positions_seen.append(total_seen)
            position_seen_prop.append(total_seen / (num_items * num_lists))
            sa_count.append(sa)
            ra_count.append(ra)
            r_value.append(r)
            printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method,
                     wall_time,
                     total_positions_seen,
                     position_seen_prop, sa_count,
                     ra_count,
                     data_name, count_White, count_Black, count_Asian, count_AmIn, count_Other, r_value)



    #DIVTOPK
    for r_cnt in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        fairness_string = 'rooney ' + str(r_cnt)
        r = r_cnt
        delta = 0
        times = []
        for t in range(0, run_cnt):
            start_time = time.time()
            K_items_FF, K_scores_FF = baseline_divtopk(fairness_string, delta, L_items, L_scores, candidate_db, k)
            end_time = time.time()
            times.append(end_time - start_time)
        set_groups = np.asarray([candidate_db[1, item] for item in K_items_FF])
        subset.append(K_items_FF)
        subset_scores.append(K_scores_FF)
        fairness_goal.append(fairness_string)
        utility_ratio.append(np.sum(K_scores_FF) / max_util)
        method.append('divtopk')
        wall_time.append(np.mean(times))
        data_name.append(dataset)
        grp_cnt = count_grp_members(candidate_db[1, :], set_groups)
        count_White.append(grp_cnt[0])
        count_Black.append(grp_cnt[1])
        count_Asian.append(grp_cnt[2])
        count_AmIn.append(grp_cnt[3])
        count_Other.append(grp_cnt[4])
        #_, _, sa, ra, total_seen = baseline_divtopk_perfcounts(fairness_string, delta, L_items, L_scores, candidate_db, k)
        total_positions_seen.append(num_items * num_lists)
        position_seen_prop.append(num_items * num_lists / (num_items * num_lists))
        sa_count.append(num_items)
        ra_count.append(0)
        r_value.append(r)
        printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method,
                 wall_time,
                 total_positions_seen,
                 position_seen_prop, sa_count,
                 ra_count,
                 data_name, count_White, count_Black, count_Asian, count_AmIn, count_Other, r_value)

    #Gupta
    for r_cnt in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        fairness_string = 'rooney ' + str(r_cnt)
        r = r_cnt
        DDP_val = 0.05
        times = []
        for t in range(0, run_cnt):
            start_time = time.time()
            K_items_FF, K_scores_FF, avg_exp = baseline_FGS(DDP_val, L_items, L_scores, candidate_db, k)
            end_time = time.time()
            times.append(end_time - start_time)
        set_groups = np.asarray([candidate_db[1, item] for item in K_items_FF])
        subset.append(K_items_FF)
        subset_scores.append(K_scores_FF)
        fairness_goal.append(fairness_string)
        utility_ratio.append(np.sum(K_scores_FF) / max_util)
        method.append('fair-exposure')
        wall_time.append(np.mean(times))
        data_name.append(dataset)
        grp_cnt = count_grp_members(candidate_db[1, :], set_groups)
        count_White.append(grp_cnt[0])
        count_Black.append(grp_cnt[1])
        count_Asian.append(grp_cnt[2])
        count_AmIn.append(grp_cnt[3])
        count_Other.append(grp_cnt[4])
        #_, _, sa, ra, total_seen = baseline_FGS_perfcounts(DDP_val, L_items, L_scores, candidate_db, k)
        total_positions_seen.append(num_items * num_lists)
        position_seen_prop.append(num_items * num_lists / (num_items * num_lists))
        sa_count.append(num_items)
        ra_count.append(0)
        r_value.append(r)
        printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method,
                 wall_time,
                 total_positions_seen,
                 position_seen_prop, sa_count,
                 ra_count,
                 data_name, count_White, count_Black, count_Asian, count_AmIn, count_Other, r_value)

    # Feng
    for r_cnt in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        fairness_string = 'rooney ' + str(r_cnt)
        r = r_cnt
        epsilon = 0.6
        times = []
        for t in range(0, run_cnt):
            seed = t
            start_time = time.time()
            K_items_FF, K_scores_FF= baseline_fengetal(epsilon, L_items, L_scores, candidate_db, k, seed)
            end_time = time.time()
            times.append(end_time - start_time)
        set_groups = np.asarray([candidate_db[1, item] for item in K_items_FF])

        subset.append(K_items_FF)
        subset_scores.append(K_scores_FF)
        fairness_goal.append(fairness_string)
        utility_ratio.append(np.sum(K_scores_FF) / max_util)
        method.append('fair-epsilon-greedy')
        wall_time.append(np.mean(times))
        data_name.append(dataset)
        grp_cnt = count_grp_members(candidate_db[1, :], set_groups)
        count_White.append(grp_cnt[0])
        count_Black.append(grp_cnt[1])
        count_Asian.append(grp_cnt[2])
        count_AmIn.append(grp_cnt[3])
        count_Other.append(grp_cnt[4])
        total_positions_seen.append(num_items * num_lists)
        position_seen_prop.append(num_items * num_lists / (num_items * num_lists))
        sa_count.append(num_items)
        ra_count.append(0)
        r_value.append(r)
        printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method,
                 wall_time,
                 total_positions_seen,
                 position_seen_prop, sa_count,
                 ra_count,
                 data_name, count_White, count_Black, count_Asian, count_AmIn, count_Other, r_value)

k = 100
iter = 1
data = 'adult'
execute(data, k, iter, 'rooney_rtask'+ data +'.csv')