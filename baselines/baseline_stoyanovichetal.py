""" Implementation of Algorithm 1 from online set selection with fairness and diversity constraints
    ----------
    References:
    Stoyanovich, J., Yang, K., & Jagadish, H. V. (2018, January).
    Online set selection with fairness and diversity constraints.
    In Proceedings of the EDBT Conference.
"""
import numpy as np
from baselines.aggscores import sorted_aggscores
from src import fairness_calibrator, rooney_calibrator


def baseline_divtopk(fairness, delta, L_items, L_scores, candidate_db, k):
    """
    DivTopK baseline algorithm adapted for fair multi-criteria selection.
    :param fairness: String either 'proportional', 'equal' or 'rooney <x>', where <x> is an int.
    :param delta: Float fairness-utility parameter [0,1], where 0 is strict fairness and 1 is utility maximizing.
    :param L_items: A numpy array (shape n x m) of the items in the sorted list.
    :param L_scores: A numpy array (shape n x m) of the items' scores in the sorted list.
    :param candidate_db: A numpy array (shape 2 x n) of each item's group membership,
                    first row is item ids, second row is item group membership.
    :param k: Int number of items in subset.
    :return: K_items: A numpy array of items in selected subset, K_scores: A numy array of score of each item in selected subset.
    """

    if fairness == 'proportional' or fairness == 'equal':
        floor_ids, floors = fairness_calibrator(candidate_db, k, fairness, delta)
    else: #rooney
        r = int(fairness.split()[1])
        floor_ids, floors = rooney_calibrator(candidate_db, r)

    #get aggregate scores
    pool_items, pool_scores = sorted_aggscores(L_items, L_scores)


    pool_groups = np.asarray([candidate_db[1, item] for item in pool_items])
    num_groups = np.unique(pool_groups).shape[0]
    group_lb_ub_key = np.zeros((num_groups,3), dtype = int)
    group_lb_ub_key[:, 0] = floor_ids
    group_lb_ub_key[:, 1] = floors
    group_lb_ub_key[:, 2] = np.repeat(k,num_groups)
    #pool_items, pool_scores, pool_groups, k, group_lb_ub_key
    #initialize outputs
    Topk_scores_list = []
    Topk_items_list = []
    Topk_groups_list = []
    num_groups = np.unique(pool_groups).shape[0]
    count_of_each_grp_cat = np.full((num_groups), 0, dtype=int)
    slack = k -np.sum(group_lb_ub_key[:,1])
    indx_itr = 0
    while len(Topk_groups_list) < k:
        x = pool_items[indx_itr]
        grp = pool_groups[indx_itr]
        k_i = count_of_each_grp_cat[grp]
        if  k_i < group_lb_ub_key[grp,1]:
            Topk_items_list.append(x)
            Topk_scores_list.append(pool_scores[indx_itr])
            Topk_groups_list.append(grp)
            count_of_each_grp_cat[grp] += 1
        else:
            if k_i < group_lb_ub_key[grp,2] and slack > 0:
                Topk_items_list.append(x)
                Topk_scores_list.append(pool_scores[indx_itr])
                Topk_groups_list.append(grp)
                count_of_each_grp_cat[grp] += 1
                slack -= 1
        indx_itr += 1
    K_items = np.asarray(Topk_items_list)
    K_scores = np.asarray(Topk_scores_list)

    return K_items, K_scores


def baseline_divtopk_perfcounts(fairness, delta, L_items, L_scores, candidate_db, k):
    """
    DivTopK baseline algorithm adapted for fair multi-criteria selection with performance metrics.
    :param fairness: String either 'proportional', 'equal' or 'rooney <x>', where <x> is an int.
    :param delta: Float fairness-utility parameter [0,1], where 0 is strict fairness and 1 is utility maximizing.
    :param L_items: A numpy array (shape n x m) of the items in the sorted list.
    :param L_scores: A numpy array (shape n x m) of the items' scores in the sorted list.
    :param candidate_db: A numpy array (shape 2 x n) of each item's group membership,
                    first row is item ids, second row is item group membership.
    :param k: Int number of items in subset.
    :return: K_items: A numpy array of items in selected subset, K_scores: A numy array of score of each item in selected subset,
                    sa_count: Int count of sorted access performed, ra_count: Int count of random access performed,
                    total_seen_positions: count of total positions seen.
    """

    if fairness == 'proportional' or fairness == 'equal':
        floor_ids, floors = fairness_calibrator(candidate_db, k, fairness, delta)
    else: #rooney
        r = int(fairness.split()[1])
        floor_ids, floors = rooney_calibrator(candidate_db, r)

    #get aggregate scores
    pool_items, pool_scores = sorted_aggscores(L_items, L_scores)

    num_items, num_lists = np.shape(L_items)
    pool_groups = np.asarray([candidate_db[1, 0] for item in pool_items])
    num_groups = np.unique(pool_groups).shape[0]
    group_lb_ub_key = np.zeros((num_groups,3), dtype = int)
    group_lb_ub_key[:, 0] = floor_ids
    group_lb_ub_key[:, 1] = floors
    group_lb_ub_key[:, 2] = np.repeat(k,num_groups)
    #pool_items, pool_scores, pool_groups, k, group_lb_ub_key
    #initialize outputs
    Topk_scores_list = []
    Topk_items_list = []
    Topk_groups_list = []
    num_groups = np.unique(pool_groups).shape[0]
    count_of_each_grp_cat = np.full((num_groups), 0, dtype=int)
    slack = k -np.sum(group_lb_ub_key[:,1])
    indx_itr = 0
    while len(Topk_groups_list) < k:
        x = pool_items[indx_itr]
        grp = pool_groups[indx_itr]
        k_i = count_of_each_grp_cat[grp]
        if  k_i < group_lb_ub_key[grp,1]:
            Topk_items_list.append(x)
            Topk_scores_list.append(pool_scores[indx_itr])
            Topk_groups_list.append(grp)
            count_of_each_grp_cat[grp] += 1
        else:
            if k_i < group_lb_ub_key[grp,2] and slack > 0:
                Topk_items_list.append(x)
                Topk_scores_list.append(pool_scores[indx_itr])
                Topk_groups_list.append(grp)
                count_of_each_grp_cat[grp] += 1
                slack -= 1
        indx_itr += 1
    K_items = np.asarray(Topk_items_list)
    K_scores = np.asarray(Topk_scores_list)
    total_seen_positions = num_lists * num_items

    sa_count = num_items
    ra_count = 0

    return K_items, K_scores, sa_count, ra_count, total_seen_positions


