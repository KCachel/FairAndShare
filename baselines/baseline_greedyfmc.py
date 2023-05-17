import numpy as np

from src import fairness_calibrator, rooney_calibrator



def greedyFMC(fairness, delta, L_items, L_scores, candidate_db, k):
    """
    Greedy Fair Algorithm for Multi-Criteria Selection.
    :param fairness: String either 'proportional', 'equal' or 'rooney <x>', where <x> is an int.
    :param delta: Float fairness-utility parameter [0,1], where 0 is strict fairness and 1 is utility maximizing.
    :param L_items: A numpy array (shape n x m) of the items in the sorted list.
    :param L_scores: A numpy array (shape n x m) of the items' scores in the sorted list.
    :param candidate_db: A numpy array (shape 2 x n) of each item's group membership,
                    first row is item ids, second row is item group membership
    :param k: Int number of items in subset
    :param t_style: String either 'TA' for Threshold access style, 'BPA' for BPA access style, or 'BPA2' for BPA2 access.
    :return: K_items: A numpy array of items in selected subset, K_scores: A numy array of score of each item in selected subset,
                    sa_count: Int count of sorted access performed, ra_count: Int count of random access performed,
                    total_seen_positions: count of total positions seen.
    """


    if fairness == 'proportional' or fairness == 'equal':
        floor_ids, floors = fairness_calibrator(candidate_db, k, fairness, delta)
    else: #rooney
        r = int(fairness.split()[1])
        floor_ids, floors = rooney_calibrator(candidate_db, r)

    num_items, num_lists = np.shape(L_items)
    seen_positions = np.zeros_like(L_items) #positions , then lists

    row_pos = 0

    K_items = []
    K_scores = []
    while np.sum(floors) > 0:
        items_at_pos = L_items[row_pos, :]
        scores_at_pos = L_scores[row_pos, :] #sorted access
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
            group_id = candidate_db[1, seen_item]
            if seen_item not in K_items and len(K_items) < k and floors[np.where(floor_ids == group_id)[0][0]] > 0: #add it
                K_items.append(seen_item)
                K_scores.append(seen_item_score)
                floors[np.where(floor_ids == group_id)[0][0]] -= 1
        row_pos += 1

    #all group constraints satisfied
    if len(K_items) < k:
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
    K_items = K_items.astype(int, copy=False)
    return K_items, K_scores


def greedyFMC_perfcounts(fairness, delta, L_items, L_scores, candidate_db, k):
    """
    Greedy Fair Algorithm for Multi-Criteria Selection.
    :param fairness: String either 'proportional', 'equal' or 'rooney <x>', where <x> is an int.
    :param delta: Float fairness-utility parameter [0,1], where 0 is strict fairness and 1 is utility maximizing.
    :param L_items: A numpy array (shape n x m) of the items in the sorted list.
    :param L_scores: A numpy array (shape n x m) of the items' scores in the sorted list.
    :param candidate_db: A numpy array (shape 2 x n) of each item's group membership,
                    first row is item ids, second row is item group membership
    :param k: Int number of items in subset
    :param t_style: String either 'TA' for Threshold access style, 'BPA' for BPA access style, or 'BPA2' for BPA2 access.
    :return: K_items: A numpy array of items in selected subset, K_scores: A numy array of score of each item in selected subset,
                    sa_count: Int count of sorted access performed, ra_count: Int count of random access performed,
                    total_seen_positions: count of total positions seen.
    """


    if fairness == 'proportional' or fairness == 'equal':
        floor_ids, floors = fairness_calibrator(candidate_db, k, fairness, delta)
    else: #rooney
        r = int(fairness.split()[1])
        floor_ids, floors = rooney_calibrator(candidate_db, r)

    sa_count = 0
    ra_count = 0
    num_items, num_lists = np.shape(L_items)
    seen_positions = np.zeros_like(L_items) #positions , then lists

    row_pos = 0

    K_items = []
    K_scores = []
    while np.sum(floors) > 0:
        items_at_pos = L_items[row_pos, :]
        scores_at_pos = L_scores[row_pos, :] #sorted access
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
            group_id = candidate_db[1, seen_item]
            if seen_item not in K_items and len(K_items) < k and floors[np.where(floor_ids == group_id)[0][0]] > 0: #add it
                K_items.append(seen_item)
                K_scores.append(seen_item_score)
                floors[np.where(floor_ids == group_id)[0][0]] -= 1
        row_pos += 1

    #all group constraints satisfied
    if len(K_items) < k:
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

    K_items = np.asarray(K_items)
    K_scores = np.asarray(K_scores)
    total_seen_positions = np.count_nonzero(seen_positions)
    K_items = K_items.astype(int, copy=False)
    return K_items, K_scores,  sa_count, ra_count, total_seen_positions


