import copy

import numpy as np
from baselines.aggscores import sorted_aggscores

def DDP(ranking, group_ids):
    """
    Function to calculate the DDP metric value.
    :param ranking: A numpy array of item ids.
    :param group_ids: A numpy array of item group ids.
    :return: DDP: DDP metric value, avg_exp_grp: A numpy array of the average group exposure per group id.
    """

    unique_grps, grp_count_items = np.unique(group_ids, return_counts=True)
    num_items = len(ranking)
    exp_vals = exp_at_position_array(num_items)
    grp_exposures = np.zeros_like(unique_grps, dtype=np.float64)
    for i in range(0,num_items):
        grp_of_item = group_ids[i]
        exp_of_item = exp_vals[i]
        #update total group exp
        grp_exposures[grp_of_item] += exp_of_item

    avg_exp_grp = grp_exposures / grp_count_items
    DDP = np.max(avg_exp_grp) - np.min(avg_exp_grp)

    return DDP, avg_exp_grp

def exp_at_position_array(num_items):
    return np.array([(1/(np.log2(i+1))) for i in range(1,num_items+1)])




def fairgreedyswap(ranking, current_group_ids, item_ids, group_ids, bnd):
    """
    Fair Greedy Swap algorithm
    :param current_ranking: List  of item ids representing a ranking.
    :param current_group_ids: List of group ids corresponding to the current_ranking.
    :param item_ids: A numpy array of the item ids.
    :param group_ids: A numpy array of the group ids corresponding to the item_ids.
    :param bnd: Desired DDP metric value.
    :return: Current_ranking: Numpy array of fair ranking with item ids, current_group_ids: Numpy array of group ids corresponding to the fair ranking.
    """
    current_ranking = copy.deepcopy(ranking)
    num_items = len(current_ranking)
    cur_ddp, avg_exps = DDP(current_ranking, current_group_ids)
    repositions = 0
    print("exposure at start:", cur_ddp)
    while( cur_ddp > bnd ):

        # Prevent infinite loops
        if repositions > ((num_items * (num_items - 1)) / 2):
            print("Try increasing the bound, FGS is stopping")
            return current_ranking, current_group_ids
            break

        Gh = np.argmax(avg_exps)
        Gl = np.argmin(avg_exps)  # group id of group with lowest avg exposure


        Gl_positions = np.argwhere(current_group_ids == Gl).flatten()
        Gh_positions = np.argwhere(current_group_ids == Gh).flatten()

        highest_ranked_Gl = np.min(Gl_positions) # lower position (np.min) is HIGHER in ranking

        valid_Gh_items = Gh_positions < highest_ranked_Gl
        if np.sum(valid_Gh_items)!= 0: swapping_item_indx_Gl = highest_ranked_Gl
        else:
            Gl_counter = 1
            while np.sum(valid_Gh_items)== 0:
                next_highest_ranked_Gl = np.min(Gl_positions[Gl_counter:,])
                valid_Gh_items = Gh_positions < next_highest_ranked_Gl
                Gl_counter += 1
            swapping_item_indx_Gl = next_highest_ranked_Gl
        swapping_item_indx_Gh = np.max(Gh_positions[valid_Gh_items]) # higher position (npmax) is LOWER in ranking


        l = current_ranking[swapping_item_indx_Gl]
        h = current_ranking[swapping_item_indx_Gh]

        # swap
        current_ranking[swapping_item_indx_Gl] = h
        current_ranking[swapping_item_indx_Gh] = l

        repositions += 1
        #update group ids
        current_group_ids = [group_ids[np.argwhere(item_ids == i)[0][0]] for i in current_ranking]
        #set up next loop
        cur_ddp, avg_exps = DDP(current_ranking, current_group_ids)
        print("exposure after swap:", cur_ddp)

    cur_ddp, avg_exps = DDP(current_ranking, current_group_ids)
    return np.asarray(current_ranking), np.asarray(current_group_ids), avg_exps


def baseline_FGS(DDP_val, L_items, L_scores, candidate_db, k):
    """
    FGS baseline algorithm adapted for fair multi-criteria selection.
    :param DDP_val: Float desired DDP value in resulting ranking.
    :param L_items: A numpy array (shape n x m) of the items in the sorted list.
    :param L_scores: A numpy array (shape n x m) of the items' scores in the sorted list.
    :param candidate_db: A numpy array (shape 2 x n) of each item's group membership,
                    first row is item ids, second row is item group membership.
    :param k: Int number of items in subset.
    :return: K_items: A numpy array of items in selected subset, K_scores: A numy array of score of each item in selected subset,
                    sa_count: Int count of sorted access performed, ra_count: Int count of random access performed,
                    total_seen_positions: count of total positions seen.
    """


    items_sorted, scores_sorted = sorted_aggscores(L_items, L_scores)

    item_groups = np.asarray([candidate_db[1, item] for item in items_sorted])


    reranking, group_ids, avg_exps = fairgreedyswap(np.asarray(items_sorted), item_groups, candidate_db[:,0], candidate_db[:,1], DDP_val)

    #reranking_scores = np.asarray([scores_sorted[np.argwhere(np.asarray(items_sorted) == item).flatten()[0]] for item in reranking])

    K_items = reranking[0:k]
    #K_scores = reranking_scores[0:k]
    K_scores = np.asarray(
        [scores_sorted[np.argwhere(np.asarray(items_sorted) == item).flatten()[0]] for item in K_items])

    return K_items, K_scores, avg_exps

def baseline_FGS_perfcounts(DDP_val, L_items, L_scores, candidate_db, k):
    """
    FGS baseline algorithm adapted for fair multi-criteria selection with performance metrics.
    :param DDP_val: Float desired DDP value in resulting ranking.
    :param L_items: A numpy array (shape n x m) of the items in the sorted list.
    :param L_scores: A numpy array (shape n x m) of the items' scores in the sorted list.
    :param candidate_db: A numpy array (shape 2 x n) of each item's group membership,
                    first row is item ids, second row is item group membership.
    :param k: Int number of items in subset.
    :return: K_items: A numpy array of items in selected subset, K_scores: A numy array of score of each item in selected subset,
                    sa_count: Int count of sorted access performed, ra_count: Int count of random access performed,
                    total_seen_positions: count of total positions seen.
    """


    items_sorted, scores_sorted = sorted_aggscores(L_items, L_scores)
    num_items, num_lists = np.shape(L_items)

    item_groups = np.asarray([candidate_db[1, item] for item in items_sorted])


    reranking, group_ids, avg_exps = fairgreedyswap(np.asarray(items_sorted), item_groups, candidate_db[:,0], candidate_db[:,1], DDP_val)

    # reranking_scores = np.asarray([scores_sorted[np.argwhere(np.asarray(items_sorted) == item).flatten()[0]] for item in reranking])
    #
    K_items = reranking[0:k]
    # K_scores = reranking_scores[0:k]
    K_scores = np.asarray([scores_sorted[np.argwhere(np.asarray(items_sorted) == item).flatten()[0]] for item in K_items])
    total_seen_positions = num_lists * num_items
    sa_count = num_items
    ra_count = 0

    return K_items, K_scores, sa_count, ra_count, total_seen_positions


 #
