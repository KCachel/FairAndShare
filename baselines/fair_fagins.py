import numpy as np
from heapq import nlargest
from src import fairness_criteria, rooney_calibrator


def check_FA_stop(seen_items, floor_ids, floors, k, num_lists, slack):
    """
    Helper function to determine if the modified Fagins Algorithm can terminate.
    :param seen_items: A numpy tensor (shape # group x # row x # list) indicating what items have been seen.
    :param floor_ids: floor_ids: A numpy array of group ids.
    :param floors: A numpy array indexed by group ids in floor_ids containing
                    the lower bound cardinality of each group.
    :param k: Int number of items in the selected subset.
    :param num_lists: Int number of feature lists in the problem.
    :param slack: Int number of spaces in subset that are not constrained to belong to specific groups.
    :return: Boolean value denoting if FFA can terminate.
    """
    flattened_seen_items = np.sum(seen_items, axis=2)
    if slack == 0: #need to only check the groups
        for g in floor_ids:
            num_g = floors[g]
            if np.count_nonzero(flattened_seen_items[g] == num_lists) < num_g:
                return False
        return True

    if slack > 0: # need to check groups and slack pool
        for g in floor_ids: #groups
            num_g = floors[g]
            if np.count_nonzero(flattened_seen_items[g] == num_lists) < num_g:
                return False

        if np.count_nonzero(flattened_seen_items[-1] == num_lists) < k: #slack pool
            return False
        return True


def fairFA(fairness, delta, L_items, L_scores, candidate_db, k):
    """
    Fagins Algorithm modified for fair multi-criteria selection.
    :param fairness: String either 'proportional', 'equal' or 'rooney <x>', where <x> is an int.
    :param delta: Float fairness-utility parameter [0,1], where 0 is strict fairness and 1 is utility maximizing.
    :param L_items: A numpy array (shape n x m) of the items in the sorted list.
    :param L_scores: A numpy array (shape n x m) of the items' scores in the sorted list.
    :param candidate_db: A numpy array (shape 2 x n) of each item's group membership,
                    first row is item ids, second row is item group membership.
    :param k: Int number of items in subset.
    :return: K_items: A numpy array of items in selected subset, K_scores: A numpy array of score of each item in selected subset.
    """

    if fairness == 'proportional' or fairness == 'equal':
        floor_ids, floors = fairness_criteria(candidate_db, k, fairness, delta)
    else: #rooney
        r = int(fairness.split()[1])
        floor_ids, floors = rooney_calibrator(candidate_db, r)

    num_items, num_lists = np.shape(L_items)
    num_groups = np.shape(floors)[0]
    sum_floors = np.sum(floors)
    slack = k - sum_floors
    seen_items = np.zeros((num_groups+1,num_items,num_lists)) #group, row, list
    score_seen_items = np.zeros((num_groups+1,num_items,num_lists))
    row_pos = 0
    seen_positions = np.zeros_like(L_items)
    while check_FA_stop(seen_items, floor_ids, floors, k, num_lists, slack) == False: #keep looping
        items_at_pos = L_items[row_pos, :]
        scores_at_pos = L_scores[row_pos, :]  # sorted access
        seen_positions[row_pos, :] = 1  # update positions set for sorted access positions
        for list_i in range(0, num_lists):
            item = items_at_pos[list_i]
            grp = candidate_db[1, item]
            if slack == 0: #only update groups
                seen_items[grp][item][list_i] = 1
                score_seen_items[grp][item][list_i] = scores_at_pos[list_i]

            if slack > 0: #update groups and slack pool
                seen_items[grp][item][list_i] = 1
                score_seen_items[grp][item][list_i] = scores_at_pos[list_i]
                seen_items[-1][item][list_i] = 1
                score_seen_items[-1][item][list_i] = scores_at_pos[list_i]
        row_pos += 1



    #perform random accesses for groups
    for g in floor_ids:
        num_g = floors[g]
        seen_g_items = seen_items[g]
        flat_lists_g = np.sum(seen_items[g], axis = 1)
        items_to_ra = np.where((flat_lists_g > 0) & (flat_lists_g < num_lists))[0]
        for item in items_to_ra:
            lists_to_ra = np.where(seen_items[g][item] == 0)[0]
            for list_i in lists_to_ra:
                row_in_list = np.where(L_items[:, list_i] == item)[0][0]
                seen_positions[row_in_list, list_i] = 1  # update seen positions
                seen_items[g][item][list_i] = 1
                score = L_scores[row_in_list][list_i]
                score_seen_items[g][item][list_i] = score
                seen_g_items[item][list_i] = 1
                #add to slack pool anyway
                seen_items[-1][item][list_i] = 1
                score_seen_items[-1][item][list_i] = score



    #perform random accesses for slack pool and find elements
    if slack != 0:
        flat_lists_slack = np.sum(seen_items[-1], axis=1)
        items_to_ra = np.where((flat_lists_slack > 0) & (flat_lists_slack < num_lists))[0]
        for item in items_to_ra:
            lists_to_ra = np.where(seen_items[-1][item] == 0)[0]
            for list_i in lists_to_ra:
                row_in_list = np.where(L_items[:, list_i] == item)[0][0]
                seen_items[-1][item][list_i] = 1
                seen_positions[row_in_list, list_i] = 1  # update seen positions
                score = L_scores[row_in_list][list_i]
                score_seen_items[-1][item][list_i] = score




    #Fill sets
    K_items = np.asarray([]) #items
    K_scores = np.asarray([]) #scores
    for g in floor_ids:
        num_g = floors[g]
        if num_g > 0:
            group_elements_total_scores = np.sum(seen_items[g]*score_seen_items[g], axis = 1)
            highest_scoring_items = np.asarray(nlargest(num_g, range(len(group_elements_total_scores)), group_elements_total_scores.take))
            highest_scores = group_elements_total_scores[highest_scoring_items]
            K_items = np.append(K_items,highest_scoring_items)
            K_scores = np.append(K_scores, highest_scores)
            #remove from slack
            seen_items[-1][highest_scoring_items] = 0

    if slack != 0:
        #fill rest
        slack_elements_total_scores = np.sum(seen_items[-1] * score_seen_items[-1], axis=1)
        highest_scoring_slack = np.asarray(
            nlargest(slack, range(len(slack_elements_total_scores)), slack_elements_total_scores.take))
        highest_scores_slack = slack_elements_total_scores[highest_scoring_slack]
        K_items = np.append(K_items, highest_scoring_slack)
        K_scores = np.append(K_scores, highest_scores_slack)

    K_items = K_items.astype(int, copy=False)
    return K_items, K_scores


def fairFA_perfcounts(fairness, delta, L_items, L_scores, candidate_db, k):
    """
    Fagins Algorithm modified for fair multi-criteria selection with performance metrics.
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
        floor_ids, floors = fairness_criteria(candidate_db, k, fairness, delta)
    else: #rooney
        r = int(fairness.split()[1])
        floor_ids, floors = rooney_calibrator(candidate_db, r)
    sa_count = 0
    ra_count = 0
    num_items, num_lists = np.shape(L_items)
    num_groups = np.shape(floors)[0]
    sum_floors = np.sum(floors)
    slack = k - sum_floors
    seen_items = np.zeros((num_groups+1,num_items,num_lists)) #group, row, list
    score_seen_items = np.zeros((num_groups+1,num_items,num_lists))
    row_pos = 0
    seen_positions = np.zeros_like(L_items)
    while check_FA_stop(seen_items, floor_ids, floors, k, num_lists, slack) == False: #keep looping
        items_at_pos = L_items[row_pos, :]
        scores_at_pos = L_scores[row_pos, :]  # sorted access
        seen_positions[row_pos, :] = 1  # update positions set for sorted access positions
        sa_count += 1
        for list_i in range(0, num_lists):
            item = items_at_pos[list_i]
            grp = candidate_db[1, item]
            if slack == 0: #only update groups
                seen_items[grp][item][list_i] = 1
                score_seen_items[grp][item][list_i] = scores_at_pos[list_i]

            if slack > 0: #update groups and slack pool
                seen_items[grp][item][list_i] = 1
                score_seen_items[grp][item][list_i] = scores_at_pos[list_i]
                seen_items[-1][item][list_i] = 1
                score_seen_items[-1][item][list_i] = scores_at_pos[list_i]
        row_pos += 1



    #perform random accesses for groups
    for g in floor_ids:
        num_g = floors[g]
        seen_g_items = seen_items[g]
        flat_lists_g = np.sum(seen_items[g], axis = 1)
        items_to_ra = np.where((flat_lists_g > 0) & (flat_lists_g < num_lists))[0]
        for item in items_to_ra:
            lists_to_ra = np.where(seen_items[g][item] == 0)[0]
            for list_i in lists_to_ra:
                row_in_list = np.where(L_items[:, list_i] == item)[0][0]
                seen_positions[row_in_list, list_i] = 1  # update seen positions
                seen_items[g][item][list_i] = 1
                ra_count += 1
                score = L_scores[row_in_list][list_i]
                score_seen_items[g][item][list_i] = score
                seen_g_items[item][list_i] = 1
                #add to slack pool anyway
                seen_items[-1][item][list_i] = 1
                score_seen_items[-1][item][list_i] = score



    #perform random accesses for slack pool and find elements
    if slack != 0:
        flat_lists_slack = np.sum(seen_items[-1], axis=1)
        items_to_ra = np.where((flat_lists_slack > 0) & (flat_lists_slack < num_lists))[0]
        for item in items_to_ra:
            lists_to_ra = np.where(seen_items[-1][item] == 0)[0]
            for list_i in lists_to_ra:
                row_in_list = np.where(L_items[:, list_i] == item)[0][0]
                seen_items[-1][item][list_i] = 1
                seen_positions[row_in_list, list_i] = 1  # update seen positions
                score = L_scores[row_in_list][list_i]
                ra_count += 1
                #seen_items[-1][item][list_i] = 1
                score_seen_items[-1][item][list_i] = score




    #Fill sets
    K_items = np.asarray([]) #items
    K_scores = np.asarray([]) #scores
    for g in floor_ids:
        num_g = floors[g]
        if num_g > 0:
            group_elements_total_scores = np.sum(seen_items[g]*score_seen_items[g], axis = 1)
            highest_scoring_items = np.asarray(nlargest(num_g, range(len(group_elements_total_scores)), group_elements_total_scores.take))
            highest_scores = group_elements_total_scores[highest_scoring_items]
            K_items = np.append(K_items,highest_scoring_items)
            K_scores = np.append(K_scores, highest_scores)
            #remove from slack
            seen_items[-1][highest_scoring_items] = 0

    if slack != 0:
        #fill rest
        slack_elements_total_scores = np.sum(seen_items[-1] * score_seen_items[-1], axis=1)
        highest_scoring_slack = np.asarray(
            nlargest(slack, range(len(slack_elements_total_scores)), slack_elements_total_scores.take))
        highest_scores_slack = slack_elements_total_scores[highest_scoring_slack]
        K_items = np.append(K_items, highest_scoring_slack)
        K_scores = np.append(K_scores, highest_scores_slack)

    total_seen_positions = np.count_nonzero(seen_positions)
    K_items = K_items.astype(int, copy=False)
    return K_items, K_scores, sa_count, ra_count, total_seen_positions


