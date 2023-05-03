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


def GBG_fagin(fairness, delta, L_items, L_scores, candidate_db, k):
    """
    GBG fairfagin baseline.
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
    else:  # rooney
        r = int(fairness.split()[1])
        floor_ids, floors = rooney_calibrator(candidate_db, r)

    K_items = []
    K_scores = []

    for i in range(0,len(floors)):
        floors_i = np.zeros_like(floors)
        floors_i[i] = floors[i]
        k_ = floors[i]
        K_items_i, K_scores_i, _, _, _, _ = fagin_core(L_items, L_scores,floor_ids, candidate_db, floors_i, k_)
        K_items.extend(K_items_i)
        K_scores.extend(K_scores_i)

    if len(K_items) < k: #FILL REST
        row_pos = 0
        seen_positions = np.zeros_like(L_items)  # positions , then lists
        num_items, num_lists = np.shape(L_items)
        while len(K_items) < k:
            items_at_pos = L_items[row_pos, :]
            scores_at_pos = L_scores[row_pos, :]  # sorted access
            unique = np.unique(items_at_pos, return_index=True)
            unique_items = unique[0]
            seen_positions[row_pos, np.arange(0, num_lists)] = 1  # update positions set for sorted access positions
            # random access
            total_score_unique_items = []
            for i in range(0, unique_items.shape[0]):
                scores_indxs = np.where(L_items[row_pos + 1:num_items, :] == unique_items[i])
                updated_rows = scores_indxs[0] + (row_pos + 1)  # update indexes
                ra_lists = scores_indxs[1]
                seen_positions[updated_rows, ra_lists] = 1  # update seen positions
                total_score_unique_items.append(np.sum(L_scores[updated_rows, scores_indxs[1]]) + np.sum(
                    scores_at_pos[np.where(items_at_pos == unique_items[i])]))
            for seen_item_id in range(0, len(unique_items)):
                seen_item = unique_items[seen_item_id]
                seen_item_score = total_score_unique_items[seen_item_id]
                if seen_item not in K_items and len(K_items) < k:  # add it
                    K_items.append(seen_item)
                    K_scores.append(seen_item_score)
            row_pos += 1

    K_items = np.asarray(K_items)
    K_scores = np.asarray(K_scores)
    return K_items, K_scores


def GBG_fagin_perfcounts(fairness, delta, L_items, L_scores, candidate_db, k):
    """
    GBG fairfagin baseline.
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
    else:  # rooney
        r = int(fairness.split()[1])
        floor_ids, floors = rooney_calibrator(candidate_db, r)

    K_items = []
    K_scores = []
    sa_count = 0
    ra_count = 0
    total_seen_positions = 0
    seen_positions = np.zeros_like(L_items)
    for i in range(0,len(floors)):
        floors_i = np.zeros_like(floors)
        floors_i[i] = floors[i]
        k_ = floors[i]
        K_items_i, K_scores_i, sa_count_i, ra_count_i, total_seen_positions_i, seen_pos_i = fagin_core(L_items, L_scores,floor_ids, candidate_db, floors_i, k_)
        seen_positions += seen_pos_i
        K_items.extend(K_items_i)
        K_scores.extend(K_scores_i)
        sa_count += sa_count_i
        ra_count += ra_count_i
    if len(K_items) < k: #FILL REST
        row_pos = 0
        seen_positions = np.zeros_like(L_items)  # positions , then lists
        num_items, num_lists = np.shape(L_items)
        while len(K_items) < k:
            items_at_pos = L_items[row_pos, :]
            scores_at_pos = L_scores[row_pos, :]  # sorted access
            sa_count += 1
            unique = np.unique(items_at_pos, return_index=True)
            unique_items = unique[0]
            seen_positions[row_pos, np.arange(0, num_lists)] = 1  # update positions set for sorted access positions
            # random access
            total_score_unique_items = []
            for i in range(0, unique_items.shape[0]):
                scores_indxs = np.where(L_items[row_pos + 1:num_items, :] == unique_items[i])
                updated_rows = scores_indxs[0] + (row_pos + 1)  # update indexes
                ra_lists = scores_indxs[1]
                ra_count += len(ra_lists)
                seen_positions[updated_rows, ra_lists] = 1  # update seen positions
                total_score_unique_items.append(np.sum(L_scores[updated_rows, scores_indxs[1]]) + np.sum(
                    scores_at_pos[np.where(items_at_pos == unique_items[i])]))
            for seen_item_id in range(0, len(unique_items)):
                seen_item = unique_items[seen_item_id]
                seen_item_score = total_score_unique_items[seen_item_id]
                if seen_item not in K_items and len(K_items) < k:  # add it
                    K_items.append(seen_item)
                    K_scores.append(seen_item_score)
            row_pos += 1
    total_seen_positions = np.count_nonzero(seen_positions)
    K_items = np.asarray(K_items)
    K_scores = np.asarray(K_scores)
    return K_items, K_scores, sa_count, ra_count, total_seen_positions


def fagin_core(L_items, L_scores,floor_ids, candidate_db, floors, k):
    sa_count = 0
    ra_count = 0
    num_items, num_lists = np.shape(L_items)
    num_groups = np.shape(floors)[0]
    sum_floors = np.sum(floors)
    slack = k - sum_floors
    seen_items = np.zeros((num_groups + 1, num_items, num_lists))  # group, row, list
    score_seen_items = np.zeros((num_groups + 1, num_items, num_lists))
    row_pos = 0

    seen_positions = np.zeros_like(L_items)
    while check_FA_stop(seen_items, floor_ids, floors, k, num_lists, slack) == False:  # keep looping
        items_at_pos = L_items[row_pos, :]
        scores_at_pos = L_scores[row_pos, :]  # sorted access
        seen_positions[row_pos, :] = 1  # update positions set for sorted access positions
        sa_count += 1
        for list_i in range(0, num_lists):
            item = items_at_pos[list_i]
            grp = candidate_db[1, item]
            if slack == 0:  # only update groups
                seen_items[grp][item][list_i] = 1
                score_seen_items[grp][item][list_i] = scores_at_pos[list_i]

            if slack > 0:  # update groups and slack pool
                seen_items[grp][item][list_i] = 1
                score_seen_items[grp][item][list_i] = scores_at_pos[list_i]
                seen_items[-1][item][list_i] = 1
                score_seen_items[-1][item][list_i] = scores_at_pos[list_i]
        row_pos += 1

    # perform random accesses for groups
    for g in floor_ids:
        num_g = floors[g]
        seen_g_items = seen_items[g]
        flat_lists_g = np.sum(seen_items[g], axis=1)
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
                # add to slack pool anyway
                seen_items[-1][item][list_i] = 1
                score_seen_items[-1][item][list_i] = score

    # perform random accesses for slack pool and find elements
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
                # seen_items[-1][item][list_i] = 1
                score_seen_items[-1][item][list_i] = score

    # Fill sets
    K_items = np.asarray([])  # items
    K_scores = np.asarray([])  # scores
    for g in floor_ids:
        num_g = floors[g]
        if num_g > 0:
            group_elements_total_scores = np.sum(seen_items[g] * score_seen_items[g], axis=1)
            highest_scoring_items = np.asarray(
                nlargest(num_g, range(len(group_elements_total_scores)), group_elements_total_scores.take))
            highest_scores = group_elements_total_scores[highest_scoring_items]
            K_items = np.append(K_items, highest_scoring_items)
            K_scores = np.append(K_scores, highest_scores)
            # remove from slack
            seen_items[-1][highest_scoring_items] = 0

    if slack != 0:
        # fill rest
        slack_elements_total_scores = np.sum(seen_items[-1] * score_seen_items[-1], axis=1)
        highest_scoring_slack = np.asarray(
            nlargest(slack, range(len(slack_elements_total_scores)), slack_elements_total_scores.take))
        highest_scores_slack = slack_elements_total_scores[highest_scoring_slack]
        K_items = np.append(K_items, highest_scoring_slack)
        K_scores = np.append(K_scores, highest_scores_slack)

    total_seen_positions = np.count_nonzero(seen_positions)
    K_items = K_items.astype(int, copy=False)
    return K_items, K_scores, sa_count, ra_count, total_seen_positions, seen_positions
