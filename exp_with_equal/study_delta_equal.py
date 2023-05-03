import numpy as np
import pandas as pd
import time
from src import *
from baselines import *
from metrics import *

def printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method, fairness_ratio, wall_time, total_positions_seen,
             position_seen_prop, group_0, group_1, group_0_val, group_1_val, group_0_cnt, group_1_cnt, sa_count, ra_count,
             data_name, delta_val, group_0_avg_exp, group_1_avg_exp):
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
            'data_name': data_name,
            'group_0_avg_exp': group_0_avg_exp,
            'group_1_avg_exp': group_1_avg_exp
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
    if dataset == "bean":
        group_0_str = "BARBUNYA-BOMBAY-DERMASON"
        group_1_str = "CALI-SEKER-SIRA-HOROZ"
        bean_raw = pd.read_csv('Dry_Bean_Dataset.csv')
        np_candidate_ids = np.arange(0, 13611, 1, dtype=int)
        np_candidate_class = np.array(bean_raw["Class"])

        class_int = []

        for val in list(np_candidate_class):
            if val == 'BARBUNYA':
                class_int.append(0)
            if val == 'CALI':
                class_int.append(1)
            if val == 'BOMBAY':
                class_int.append(0)
            if val == 'DERMASON':
                class_int.append(0)
            if val == 'SEKER':
                class_int.append(1)
            if val == 'SIRA':
                class_int.append(1)
            if val == 'HOROZ':
                class_int.append(1)
        groups = np.asarray(class_int)
        # group_0_str = "BARBUNYA-CALI-BOMBAY-DERMASON"
        # group_1_str = "SEKER-SIRA-HOROZ"
        # bean_raw = pd.read_excel('Dry_Bean_Dataset.xlsx')
        #
        # np_candidate_ids = np.arange(0, 13611, 1, dtype=int)
        # np_candidate_class = np.array(bean_raw["Class"])
        #
        # class_int = []
        # # for val in list(np_candidate_class):
        # #     if val == 'BARBUNYA':
        # #         class_int.append(0)
        # #     if val == 'CALI':
        # #         class_int.append(0)
        # #     if val == 'BOMBAY':
        # #         class_int.append(0)
        # #     if val == 'DERMASON':
        # #         class_int.append(0)
        # #     if val == 'SEKER':
        # #         class_int.append(1)
        # #     if val == 'SIRA':
        # #         class_int.append(1)
        # #     if val == 'HOROZ':
        # #         class_int.append(1)
        # # groups = np.asarray(class_int)
        #
        # for val in list(np_candidate_class):
        #     if val == 'BARBUNYA':
        #         class_int.append(0)
        #     if val == 'CALI':
        #         class_int.append(1)
        #     if val == 'BOMBAY':
        #         class_int.append(0)
        #     if val == 'DERMASON':
        #         class_int.append(0)
        #     if val == 'SEKER':
        #         class_int.append(1)
        #     if val == 'SIRA':
        #         class_int.append(1)
        #     if val == 'HOROZ':
        #         class_int.append(1)
        # groups = np.asarray(class_int)

        Area = np.array(bean_raw["Area"])
        Area = np.column_stack((np_candidate_ids, Area))
        Area_sorted = np.flipud(Area[Area[:, 1].argsort()])

        Perimeter = np.array(bean_raw["Perimeter"])
        Perimeter = np.column_stack((np_candidate_ids, Perimeter))
        Perimeter_sorted = np.flipud(Perimeter[Perimeter[:, 1].argsort()])

        MajorAxisLength = np.array(bean_raw["MajorAxisLength"])
        MajorAxisLength = np.column_stack((np_candidate_ids, MajorAxisLength))
        MajorAxisLength_sorted = np.flipud(MajorAxisLength[MajorAxisLength[:, 1].argsort()])

        MinorAxisLength = np.array(bean_raw["MinorAxisLength"])
        MinorAxisLength = np.column_stack((np_candidate_ids, MinorAxisLength))
        MinorAxisLength_sorted = np.flipud(MinorAxisLength[MinorAxisLength[:, 1].argsort()])

        AspectRation = np.array(bean_raw["AspectRation"])
        AspectRation = np.column_stack((np_candidate_ids, AspectRation))
        AspectRation_sorted = np.flipud(AspectRation[AspectRation[:, 1].argsort()])

        Eccentricity = np.array(bean_raw["Eccentricity"])
        Eccentricity = np.column_stack((np_candidate_ids, Eccentricity))
        Eccentricity_sorted = np.flipud(Eccentricity[Eccentricity[:, 1].argsort()])

        ConvexArea = np.array(bean_raw["ConvexArea"])
        ConvexArea = np.column_stack((np_candidate_ids, ConvexArea))
        ConvexArea_sorted = np.flipud(ConvexArea[ConvexArea[:, 1].argsort()])

        EquivDiameter = np.array(bean_raw["EquivDiameter"])
        EquivDiameter = np.column_stack((np_candidate_ids, EquivDiameter))
        EquivDiameter_sorted = np.flipud(EquivDiameter[EquivDiameter[:, 1].argsort()])

        Extent = np.array(bean_raw["Extent"])
        Extent = np.column_stack((np_candidate_ids, Extent))
        Extent_sorted = np.flipud(Extent[Extent[:, 1].argsort()])

        Solidity = np.array(bean_raw["Solidity"])
        Solidity = np.column_stack((np_candidate_ids, Solidity))
        Solidity_sorted = np.flipud(Solidity[Solidity[:, 1].argsort()])

        roundness = np.array(bean_raw["roundness"])
        roundness = np.column_stack((np_candidate_ids, roundness))
        roundness_sorted = np.flipud(roundness[roundness[:, 1].argsort()])

        Compactness = np.array(bean_raw["Compactness"])
        Compactness = np.column_stack((np_candidate_ids, Compactness))
        Compactness_sorted = np.flipud(Compactness[Compactness[:, 1].argsort()])

        ShapeFactor1 = np.array(bean_raw["ShapeFactor1"])
        ShapeFactor1 = np.column_stack((np_candidate_ids, ShapeFactor1))
        ShapeFactor1_sorted = np.flipud(ShapeFactor1[ShapeFactor1[:, 1].argsort()])

        ShapeFactor2 = np.array(bean_raw["ShapeFactor2"])
        ShapeFactor2 = np.column_stack((np_candidate_ids, ShapeFactor2))
        ShapeFactor2_sorted = np.flipud(ShapeFactor2[ShapeFactor2[:, 1].argsort()])

        ShapeFactor3 = np.array(bean_raw["ShapeFactor3"])
        ShapeFactor3 = np.column_stack((np_candidate_ids, ShapeFactor3))
        ShapeFactor3_sorted = np.flipud(ShapeFactor3[ShapeFactor3[:, 1].argsort()])

        ShapeFactor4 = np.array(bean_raw["ShapeFactor4"])
        ShapeFactor4 = np.column_stack((np_candidate_ids, ShapeFactor4))
        ShapeFactor4_sorted = np.flipud(ShapeFactor4[ShapeFactor4[:, 1].argsort()])

        L_items = np.column_stack((Area_sorted[:, 0],
                                   Perimeter_sorted[:, 0],
                                   MajorAxisLength_sorted[:, 0],
                                   MinorAxisLength_sorted[:, 0],
                                   AspectRation_sorted[:, 0],
                                   Eccentricity_sorted[:, 0],
                                   ConvexArea_sorted[:, 0],
                                   EquivDiameter_sorted[:, 0],
                                   Extent_sorted[:, 0],
                                   Solidity_sorted[:, 0],
                                   roundness_sorted[:, 0],
                                   Compactness_sorted[:, 0],
                                   ShapeFactor1_sorted[:, 0],
                                   ShapeFactor2_sorted[:, 0],
                                   ShapeFactor3_sorted[:, 0],
                                   ShapeFactor4_sorted[:, 0]))
        L_items = np.int_(L_items)
        L_scores = np.column_stack((Area_sorted[:, 1],
                                    Perimeter_sorted[:, 1],
                                    MajorAxisLength_sorted[:, 1],
                                    MinorAxisLength_sorted[:, 1],
                                    AspectRation_sorted[:, 1],
                                    Eccentricity_sorted[:, 1],
                                    ConvexArea_sorted[:, 1],
                                    EquivDiameter_sorted[:, 1],
                                    Extent_sorted[:, 1],
                                    Solidity_sorted[:, 1],
                                    roundness_sorted[:, 1],
                                    Compactness_sorted[:, 1],
                                    ShapeFactor1_sorted[:, 1],
                                    ShapeFactor2_sorted[:, 1],
                                    ShapeFactor3_sorted[:, 1],
                                    ShapeFactor4_sorted[:, 1]))

        candidate_db = np.vstack((np_candidate_ids, groups))
        num_items, num_lists = np.shape(L_items)

    if dataset == "iit":
        group_0_str = "male"
        group_1_str = "female"
        iit_raw = pd.read_csv('IIT-JEE2009.csv')
        np_candidate_ids = np.array(iit_raw["ID"])
        np_candidate_gender = np.array(iit_raw["GENDER"])
        gender_int = np.array([0 if val == 'M' else 1 for val in list(np_candidate_gender)])
        np_candidate_math = np.array(iit_raw["math"])
        np_candidate_phys = np.array(iit_raw["phys"])
        np_candidate_chem = np.array(iit_raw["chem"])

        np_math = np.column_stack((np_candidate_ids, np_candidate_math))
        math_sorted = np.flipud(np_math[np_math[:, 1].argsort()])
        np_phys = np.column_stack((np_candidate_ids, np_candidate_phys))
        phys_sorted = np.flipud(np_phys[np_phys[:, 1].argsort()])
        np_chem = np.column_stack((np_candidate_ids, np_candidate_chem))
        chem_sorted = np.flipud(np_chem[np_chem[:, 1].argsort()])

        L_items = np.column_stack((math_sorted[:, 0], phys_sorted[:, 0], chem_sorted[:, 0]))
        L_scores = np.column_stack((math_sorted[:, 1], phys_sorted[:, 1], chem_sorted[:, 1]))
        candidate_db = np.vstack((np_candidate_ids, gender_int))
        L_scores += 160
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
    group_0_avg_exp = []
    group_1_avg_exp = []

    fairness_string = "equal"
    delta_dict = {}
    delta_dict[0] = "fair-"
    delta_dict[1] = ""
    delta_dict[.1] = "fair-"
    delta_dict[.05] = "fair-"
    np.random.seed(0)



    #Fagins
    for delta in [1, .1, .05, 0]:
        times = []
        for t in range(0, run_cnt):
            start_time = time.time()
            K_items_FF, K_scores_FF = fairFA(fairness_string, delta, L_items, L_scores, candidate_db, k)
            end_time = time.time()
            times.append(end_time - start_time)

        set_groups = np.asarray([candidate_db[1, item] for item in K_items_FF])
        proportions_of_sub, prop_ratio = balance(candidate_db[1, :], K_items_FF, set_groups)
        if delta == 1:
            MAX_UTIL = np.sum(K_scores_FF)
            protected_grp = np.argmin(proportions_of_sub)
        subset.append(K_items_FF)
        subset_scores.append(K_scores_FF)
        fairness_goal.append(fairness_string)
        utility_ratio.append(np.sum(K_scores_FF)/MAX_UTIL)
        method.append(delta_dict[delta]+'fagins')
        fairness_ratio.append(prop_ratio)
        wall_time.append(np.mean(times))
        data_name.append(dataset)
        group_0.append(group_0_str)
        group_1.append(group_1_str)
        group_0_val.append(proportions_of_sub[0])
        group_1_val.append(proportions_of_sub[1])
        group_0_avg_exp.append('n/a')
        group_1_avg_exp.append('n/a')
        _, grp_cnt = np.unique(set_groups, return_counts=True)
        group_0_cnt.append(np.count_nonzero(set_groups == 0))
        group_1_cnt.append(np.count_nonzero(set_groups == 1))
        _, _, sa, ra, total_seen = fairFA_perfcounts(fairness_string, delta, L_items, L_scores, candidate_db, k)
        total_positions_seen.append(total_seen)
        position_seen_prop.append(total_seen / (num_items * num_lists))
        sa_count.append(sa)
        ra_count.append(ra)
        delta_val.append(delta)
    printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method, fairness_ratio, wall_time,
             total_positions_seen,
             position_seen_prop, group_0, group_1, group_0_val, group_1_val, group_0_cnt, group_1_cnt, sa_count,
             ra_count,
             data_name, delta_val, group_0_avg_exp, group_1_avg_exp)

    # #Thresholds
    for t_style in ['TA','BPA', 'BPA2']:
        for delta in [1, .1, .05, 0]:
            times = []
            for t in range(0, run_cnt):
                start_time = time.time()
                K_items, K_scores = ThresholdFMCS(fairness_string, delta, L_items, L_scores, candidate_db, k, t_style)
                end_time = time.time()
                times.append(end_time - start_time)
            set_groups = np.asarray([candidate_db[1, item] for item in K_items])
            proportions_of_sub, prop_ratio = balance(candidate_db[1, :], K_items, set_groups)
            subset.append(K_items)
            subset_scores.append(K_scores)
            fairness_goal.append(fairness_string)
            utility_ratio.append(np.sum(K_scores) / MAX_UTIL)
            method.append(delta_dict[delta] + t_style.lower())
            fairness_ratio.append(prop_ratio)
            wall_time.append(np.mean(times))
            data_name.append(dataset)
            group_0.append(group_0_str)
            group_1.append(group_1_str)
            group_0_val.append(proportions_of_sub[0])
            group_1_val.append(proportions_of_sub[1])
            _, grp_cnt = np.unique(set_groups, return_counts=True)
            group_0_cnt.append(np.count_nonzero(set_groups == 0))
            group_1_cnt.append(np.count_nonzero(set_groups == 1))
            group_0_avg_exp.append('n/a')
            group_1_avg_exp.append('n/a')
            _, _, sa, ra, total_seen = ThresholdFMCS_perfcounts(fairness_string, delta, L_items, L_scores, candidate_db, k, t_style)
            total_positions_seen.append(total_seen)
            position_seen_prop.append(total_seen / (num_items * num_lists))
            sa_count.append(sa)
            ra_count.append(ra)
            delta_val.append(delta)
        printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method, fairness_ratio, wall_time,
                 total_positions_seen,
                 position_seen_prop, group_0, group_1, group_0_val, group_1_val, group_0_cnt, group_1_cnt, sa_count,
                 ra_count,
                 data_name, delta_val, group_0_avg_exp, group_1_avg_exp)

    # Greedy Fair
    for delta in [1, .1, .05, 0]:
        times = []
        for t in range(0, run_cnt):
            start_time = time.time()
            K_items, K_scores = greedyFMC(fairness_string, delta, L_items, L_scores, candidate_db, k)
            end_time = time.time()
            times.append(end_time - start_time)
        set_groups = np.asarray([candidate_db[1, item] for item in K_items])
        proportions_of_sub, prop_ratio = balance(candidate_db[1, :], K_items, set_groups)
        subset.append(K_items)
        subset_scores.append(K_scores)
        fairness_goal.append(fairness_string)
        utility_ratio.append(np.sum(K_scores) / MAX_UTIL)
        method.append(delta_dict[delta] + 'Greedy_FMC')
        fairness_ratio.append(prop_ratio)
        wall_time.append(np.mean(times))
        data_name.append(dataset)
        group_0.append(group_0_str)
        group_1.append(group_1_str)
        group_0_val.append(proportions_of_sub[0])
        group_1_val.append(proportions_of_sub[1])
        _, grp_cnt = np.unique(set_groups, return_counts=True)
        group_0_cnt.append(np.count_nonzero(set_groups == 0))
        group_1_cnt.append(np.count_nonzero(set_groups == 1))
        group_0_avg_exp.append('n/a')
        group_1_avg_exp.append('n/a')
        _, _, sa, ra, total_seen = greedyFMC_perfcounts(fairness_string, delta, L_items, L_scores, candidate_db, k)
        total_positions_seen.append(total_seen)
        position_seen_prop.append(total_seen / (num_items * num_lists))
        sa_count.append(sa)
        ra_count.append(ra)
        delta_val.append(delta)
    printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method, fairness_ratio, wall_time,
             total_positions_seen,
             position_seen_prop, group_0, group_1, group_0_val, group_1_val, group_0_cnt, group_1_cnt, sa_count,
             ra_count,
             data_name, delta_val, group_0_avg_exp, group_1_avg_exp)
    # GBG_Fagins

    for delta in [.1, .05, 0]:
        times = []
        for t in range(0, run_cnt):
            start_time = time.time()
            K_items, K_scores = GBG_fagin(fairness_string, delta, L_items, L_scores, candidate_db, k)
            end_time = time.time()
            times.append(end_time - start_time)
        set_groups = np.asarray([candidate_db[1, item] for item in K_items])
        proportions_of_sub, prop_ratio = balance(candidate_db[1, :], K_items, set_groups)
        subset.append(K_items)
        subset_scores.append(K_scores)
        fairness_goal.append(fairness_string)
        utility_ratio.append(np.sum(K_scores) / MAX_UTIL)
        method.append(delta_dict[delta] + 'GBG_Fagin')
        fairness_ratio.append(prop_ratio)
        wall_time.append(np.mean(times))
        data_name.append(dataset)
        group_0.append(group_0_str)
        group_1.append(group_1_str)
        group_0_val.append(proportions_of_sub[0])
        group_1_val.append(proportions_of_sub[1])
        _, grp_cnt = np.unique(set_groups, return_counts=True)
        group_0_cnt.append(np.count_nonzero(set_groups == 0))
        group_1_cnt.append(np.count_nonzero(set_groups == 1))
        group_0_avg_exp.append('n/a')
        group_1_avg_exp.append('n/a')
        _, _, sa, ra, total_seen = GBG_fagin_perfcounts(fairness_string, delta, L_items, L_scores, candidate_db, k)
        total_positions_seen.append(total_seen)
        position_seen_prop.append(total_seen / (num_items * num_lists))
        sa_count.append(sa)
        ra_count.append(ra)
        delta_val.append(delta)
    printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method, fairness_ratio, wall_time,
             total_positions_seen,
             position_seen_prop, group_0, group_1, group_0_val, group_1_val, group_0_cnt, group_1_cnt, sa_count,
             ra_count,
             data_name, delta_val, group_0_avg_exp, group_1_avg_exp)

    # GBG Threshold
    for t_style in ['TA', 'BPA', 'BPA2']:
        for delta in [.1, .05, 0]:
            times = []
            for t in range(0, run_cnt):
                start_time = time.time()
                K_items, K_scores = GBG_threshold(fairness_string, delta, L_items, L_scores, candidate_db, k,
                                                  t_style)
                end_time = time.time()
                times.append(end_time - start_time)
            if delta == 1: max_util = np.sum(K_scores)
            set_groups = np.asarray([candidate_db[1, item] for item in K_items])
            proportions_of_sub, prop_ratio = balance(candidate_db[1, :], K_items, set_groups)
            subset.append(K_items)
            subset_scores.append(K_scores)
            fairness_goal.append(fairness_string)
            utility_ratio.append(np.sum(K_scores) / MAX_UTIL)
            method.append(delta_dict[delta] + 'GBG_' + t_style.lower())
            fairness_ratio.append(prop_ratio)
            wall_time.append(np.mean(times))
            data_name.append(dataset)
            group_0.append(group_0_str)
            group_1.append(group_1_str)
            group_0_val.append(proportions_of_sub[0])
            group_1_val.append(proportions_of_sub[1])
            _, grp_cnt = np.unique(set_groups, return_counts=True)
            group_0_cnt.append(np.count_nonzero(set_groups == 0))
            group_1_cnt.append(np.count_nonzero(set_groups == 1))
            group_0_avg_exp.append('n/a')
            group_1_avg_exp.append('n/a')
            _, _, sa, ra, total_seen = GBG_threshold_perfcounts(fairness_string, delta, L_items, L_scores,
                                                                candidate_db,
                                                                k, t_style)
            total_positions_seen.append(total_seen)
            position_seen_prop.append(total_seen / (num_items * num_lists))
            sa_count.append(sa)
            ra_count.append(ra)
            delta_val.append(delta)
        printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method, fairness_ratio,
                 wall_time,
                 total_positions_seen,
                 position_seen_prop, group_0, group_1, group_0_val, group_1_val, group_0_cnt, group_1_cnt, sa_count,
                 ra_count,
                 data_name, delta_val, group_0_avg_exp, group_1_avg_exp)
    #DIVTOPK
    for delta in [1, .1, .05, 0]:
        times = []
        for t in range(0, run_cnt):
            start_time = time.time()
            K_items_FF, K_scores_FF = baseline_divtopk(fairness_string, delta, L_items, L_scores, candidate_db, k)
            end_time = time.time()
            times.append(end_time - start_time)
        set_groups = np.asarray([candidate_db[1, item] for item in K_items_FF])
        proportions_of_sub, prop_ratio = balance(candidate_db[1, :], K_items_FF, set_groups)
        subset.append(K_items_FF)
        subset_scores.append(K_scores_FF)
        fairness_goal.append(fairness_string)
        utility_ratio.append(np.sum(K_scores_FF) / MAX_UTIL)
        method.append('divtopk')
        fairness_ratio.append(prop_ratio)
        wall_time.append(np.mean(times))
        data_name.append(dataset)
        group_0.append(group_0_str)
        group_1.append(group_1_str)
        group_0_val.append(proportions_of_sub[0])
        group_1_val.append(proportions_of_sub[1])
        _, grp_cnt = np.unique(set_groups, return_counts=True)
        group_0_cnt.append(np.count_nonzero(set_groups == 0))
        group_1_cnt.append(np.count_nonzero(set_groups == 1))
        group_0_avg_exp.append('n/a')
        group_1_avg_exp.append('n/a')
        #_, _, sa, ra, total_seen = baseline_divtopk_perfcounts(fairness_string, delta, L_items, L_scores, candidate_db, k)
        total_positions_seen.append(num_items * num_lists)
        position_seen_prop.append(num_items * num_lists / (num_items * num_lists))
        sa_count.append(num_items)
        ra_count.append(0)
        delta_val.append(delta)
        printoff(output_file, subset, subset_scores, fairness_goal, utility_ratio, method, fairness_ratio, wall_time,
                 total_positions_seen,
                 position_seen_prop, group_0, group_1, group_0_val, group_1_val, group_0_cnt, group_1_cnt, sa_count,
                 ra_count,
                 data_name, delta_val, group_0_avg_exp, group_1_avg_exp)
iter = 5
data = 'lc'
execute(data, 100, iter, 'equal_study_delta'+ data +'.csv')
data = 'hc'
execute(data, 100, iter, 'equal_study_delta'+ data +'.csv')
data = 'gauss'
execute(data, 100, iter, 'equal_study_delta'+ data +'.csv')
data = 'bean'
execute(data, 100, iter, 'equal_study_delta'+ data +'.csv')
data = 'iit'
execute(data, 1000, iter, 'equal_study_delta'+ data +'.csv')
