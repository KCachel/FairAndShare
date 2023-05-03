import numpy as np
from heapq import nlargest
from src import fairness_criteria, rooney_calibrator


def find_bestposition_vals(bp_vals, seen_positions):
    """
    Helper function to determine the best positions for each list.
    :param bp_vals: A numpy array (shape 1 x m) of ints representing the best position in each list.
    :param seen_positions: A numpy array (shape n x m) of boolean values indicating which positions have been seen in the lists.
    :return: bp_vals: updated best positions based on what has been seen.
    """
    num_items = seen_positions.shape[0]
    for list_i in range(0, len(bp_vals)):
        bp = bp_vals[list_i]
        bp += 1
        while (bp < num_items ) and (seen_positions[bp,list_i] == 1):
            bp += 1
        bp_vals[list_i] = bp - 1
    return bp_vals

def GBG_thresholdsplit(fairness, delta, L_items, L_scores, candidate_db, k, t_style):
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
    else: #rooney
        r = int(fairness.split()[1])
        floor_ids, floors = rooney_calibrator(candidate_db, r)

    K_items = []
    K_scores = []
    if np.sum(floors) < k:
        num = k - np.sum(floors)
        div = len(floors)
        spillover = [num // div + (1 if x < num % div else 0)  for x in range (div)]
        for i in range(0, len(spillover)):
            floors[i] += spillover[i]
    for i in range(0,len(floors)):
        floors_i = np.zeros_like(floors)
        floors_i[i] = floors[i]
        k = floors[i]
        if k > 0:
            K_items_i, K_scores_i, _, _, _ , _= threshold_core(L_items, L_scores,floor_ids, candidate_db, floors_i, k, t_style)
            K_items.extend(K_items_i)
            K_scores.extend(K_scores_i)
    K_items = np.asarray(K_items)
    K_scores = np.asarray(K_scores)
    return K_items, K_scores


def GBG_thresholdsplit_perfcounts(fairness, delta, L_items, L_scores, candidate_db, k, t_style):
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
    if np.sum(floors) < k:
        num = k - np.sum(floors)
        div = len(floors)
        spillover = [num // div + (1 if x < num % div else 0) for x in range(div)]
        for i in range(0, len(spillover)):
            floors[i] += spillover[i]
    for i in range(0,len(floors)):
        floors_i = np.zeros_like(floors)
        floors_i[i] = floors[i]
        k = floors[i]
        K_items_i, K_scores_i, sa_count_i, ra_count_i, total_seen_positions_i, seen_pos_i = threshold_core(L_items, L_scores,floor_ids, candidate_db, floors_i, k, t_style)
        seen_positions += seen_pos_i
        K_items.extend(K_items_i)
        K_scores.extend(K_scores_i)
        sa_count += sa_count_i
        ra_count += ra_count_i
    total_seen_positions = np.count_nonzero(seen_positions)
    K_items = np.asarray(K_items)
    K_scores = np.asarray(K_scores)
    return K_items, K_scores, sa_count, ra_count, total_seen_positions


def threshold_core(L_items, L_scores,floor_ids, candidate_db, floors, k, t_style):
    sa_count = 0
    ra_count = 0
    num_items, num_lists = np.shape(L_items)
    num_groups = np.shape(floors)[0]
    max_floor = np.max(floors)
    S_grp_items = np.full([num_groups, max_floor], np.nan)
    S_grp_scores = np.full([num_groups, max_floor], np.nan)
    seen_positions = np.zeros_like(L_items)  # positions , then lists

    if t_style == "TA":
        threshold = np.inf
        if np.sum(floors) == k:  # Perfect fairness mode

            # Every item's score is less than the threshold and we do not have k elements keep looping
            row_pos = 0
            while not (np.count_nonzero(S_grp_scores >= threshold) == k):  # k items each >= threshold

                items_at_pos = L_items[row_pos, :]
                scores_at_pos = L_scores[row_pos, :]
                sa_count += 1
                seen_positions[row_pos, np.arange(0, num_lists)] = 1  # update positions set for sorted access positions
                unique = np.unique(items_at_pos, return_index=True)
                unique_items = unique[0]

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
                    grp_floor = floors[np.where(floor_ids == group_id)[0][0]]

                    if seen_item in S_grp_items[group_id] or grp_floor == 0:
                        pass
                    # if nothing in S or less than floor items add it
                    elif np.count_nonzero(
                            np.isfinite(S_grp_scores[group_id])) < grp_floor:  # CHANGE TO LESS THAN FLOOR ITEMS
                        indx_to_fill = np.argwhere(np.isnan(S_grp_scores[group_id]))[0]
                        S_grp_items[group_id][indx_to_fill] = seen_item
                        S_grp_scores[group_id][indx_to_fill] = seen_item_score
                    else:
                        min_score_for_grp = np.min(S_grp_scores[group_id][np.isfinite(S_grp_scores[group_id])])
                        if min_score_for_grp < seen_item_score:
                            indx_min = np.argwhere(S_grp_scores[group_id] == min_score_for_grp)[0][
                                0]  # index of min item
                            S_grp_items[group_id][indx_min] = seen_item  # replace min item
                            S_grp_scores[group_id][indx_min] = seen_item_score  # replace min item's score
                # update threshold
                threshold = np.sum(scores_at_pos)
                row_pos += 1

            indxs_of_finite = np.isfinite(S_grp_scores)
            K_scores = S_grp_scores[indxs_of_finite]
            K_items = S_grp_items[indxs_of_finite]
            total_seen_positions = np.count_nonzero(seen_positions)
            K_items = K_items.astype(int, copy=False)
            return K_items, K_scores, sa_count, ra_count, total_seen_positions, seen_positions

        if np.sum(floors) < k:  # Slack mode
            # Every item's score is less than the threshold and we do not have k elements keep looping
            # use tf check list of lists
            row_pos = 0
            sum_floors = np.sum(floors)
            slack = k - sum_floors
            S_slack_items = np.array([])  # hold slack items
            S_slack_scores = np.array([])  # hold slack items' scores
            while not (np.count_nonzero(S_slack_scores >= threshold) == slack) or not (
                    np.count_nonzero(S_grp_scores >= threshold) == sum_floors):

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
                    group_id = candidate_db[1, seen_item]
                    grp_floor = floors[np.where(floor_ids == group_id)[0][0]]

                    seen_in_group_bool = seen_item in S_grp_items[group_id]
                    seen_in_S_bool = seen_item in S_slack_items
                    if seen_in_group_bool or seen_in_S_bool:
                        pass
                    # if nothing in S_group or less than floor items add it
                    elif np.count_nonzero(
                            np.isfinite(S_grp_scores[group_id])) < grp_floor and seen_in_group_bool == False:
                        indx_to_fill = np.argwhere(np.isnan(S_grp_scores[group_id]))[0]
                        S_grp_items[group_id][indx_to_fill] = seen_item
                        S_grp_scores[group_id][indx_to_fill] = seen_item_score

                    elif seen_in_group_bool == False:
                        if S_grp_scores[group_id].shape == (0,):  # empty since floor is 0
                            min_score_for_grp = np.Inf
                        else:
                            min_score_for_grp = np.min(S_grp_scores[group_id][np.isfinite(S_grp_scores[group_id])])
                        if min_score_for_grp < seen_item_score:
                            indx_min = np.argwhere(S_grp_scores[group_id] == min_score_for_grp)[0][
                                0]  # index of min item
                            min_item = S_grp_items[group_id][indx_min]  # save min_item
                            S_grp_items[group_id][indx_min] = seen_item  # replace min item
                            S_grp_scores[group_id][indx_min] = seen_item_score  # replace min item's score
                            # Need to update set S with the min item
                            if len(S_slack_scores) < slack:  # if S has less than slack items add it
                                S_slack_items = np.append(S_slack_items, min_item)
                                S_slack_scores = np.append(S_slack_scores, min_score_for_grp)

                            else:
                                min_score = np.min(S_slack_scores)
                                if min_score < min_score_for_grp:  # lowest item in slack set is less than kicked out group item
                                    indx_minS = np.argmin(S_slack_scores)  # index of min item
                                    S_slack_items[indx_minS] = min_item  # replace min item
                                    S_slack_scores[indx_minS] = min_score_for_grp  # replace min item's score



                        # Need to update set S
                        elif len(S_slack_scores) < slack:  # if S has less than k items add it
                            S_slack_scores = np.append(S_slack_scores, seen_item_score)
                            S_slack_items = np.append(S_slack_items, seen_item)

                        elif seen_item_score > np.min(S_slack_scores):  # if item is higher than what is in slack
                            indx_minS = np.argmin(S_slack_scores)  # index of min item
                            S_slack_items[indx_minS] = seen_item  # replace min item
                            S_slack_scores[indx_minS] = seen_item_score  # replace min item's score

                # update threshold
                threshold = np.sum(scores_at_pos)
                row_pos += 1

            indxs_of_finite = np.isfinite(S_grp_scores)
            K_scores = S_grp_scores[indxs_of_finite]
            K_items = S_grp_items[indxs_of_finite]

            K_items = np.hstack((K_items, S_slack_items))
            K_scores = np.hstack((K_scores, S_slack_scores))
            total_seen_positions = np.count_nonzero(seen_positions)
            K_items = K_items.astype(int, copy=False)
            return K_items, K_scores, sa_count, ra_count, total_seen_positions, seen_positions

    elif t_style == 'BPA' or t_style == 'BPA2':

        bp_threshold = np.inf
        if t_style == 'BPA':
            bp_vals = np.zeros(num_lists, dtype=int)
        else:  # BPA2
            bp_vals = np.ones(num_lists, dtype=int) * -1

        if np.sum(floors) == k:  # Perfect fairness mode

            # Sorted Access loop
            row_pos = 0
            while not (np.count_nonzero(S_grp_scores >= bp_threshold) == k):  # k items each >= threshold
                if t_style == 'BPA':
                    items_at_pos = L_items[row_pos, :]
                    scores_at_pos = L_scores[row_pos, :]
                    seen_positions[row_pos, :] = 1  # update positions set for sorted access positions
                    sa_count += 1
                else:  # BPA2
                    bp_vals = bp_vals + 1
                    items_at_pos = L_items[bp_vals, np.arange(0, num_lists)]
                    scores_at_pos = L_scores[bp_vals, np.arange(0, num_lists)]
                    if len(np.unique(bp_vals)) == 1:  # SA
                        sa_count += 1
                    else:
                        ra_count += num_lists  # one for each list
                    seen_positions[
                        bp_vals, np.arange(0, num_lists)] = 1  # update positions set for sorted access positions

                unique = np.unique(items_at_pos, return_index=True)
                unique_items = unique[0]
                # random access
                total_score_unique_items = []
                if t_style == 'BPA':
                    for i in range(0, unique_items.shape[0]):
                        scores_indxs = np.where(L_items[row_pos + 1:num_items, :] == unique_items[i])
                        updated_rows = scores_indxs[0] + (row_pos + 1)  # update indexes
                        ra_lists = scores_indxs[1]
                        seen_positions[updated_rows, ra_lists] = 1  # update seen positions
                        ra_count += len(ra_lists)
                        total_score_unique_items.append(np.sum(L_scores[updated_rows, scores_indxs[1]]) + np.sum(
                            scores_at_pos[np.where(items_at_pos == unique_items[i])]))
                else:  # BPA2
                    for i in range(0, unique_items.shape[0]):
                        item_to_find = unique_items[i]
                        ra_lists = np.where(items_at_pos != item_to_find)[0]  # lists to search
                        updated_rows = np.asarray([], dtype=int)
                        for list_i in ra_lists:
                            row = np.where(L_items[bp_vals[list_i] + 1:num_items, list_i] == item_to_find)[0][0]
                            updated_rows = np.append(updated_rows, row + bp_vals[list_i] + 1)
                        seen_positions[updated_rows, ra_lists] = 1  # update seen positions
                        ra_count += len(ra_lists)
                        total_score_unique_items.append(np.sum(L_scores[updated_rows, ra_lists]) + np.sum(
                            scores_at_pos[np.where(items_at_pos == item_to_find)]))

                for seen_item_id in range(0, len(unique_items)):
                    seen_item = unique_items[seen_item_id]
                    seen_item_score = total_score_unique_items[seen_item_id]
                    group_id = candidate_db[1, seen_item]
                    grp_floor = floors[np.where(floor_ids == group_id)[0][0]]

                    if seen_item in S_grp_items[group_id] or grp_floor == 0:
                        pass
                    # if nothing in S or less than floor items add it
                    elif np.count_nonzero(
                            np.isfinite(S_grp_scores[group_id])) < grp_floor:  # CHANGE TO LESS THAN FLOOR ITEMS
                        indx_to_fill = np.argwhere(np.isnan(S_grp_scores[group_id]))[0]
                        S_grp_items[group_id][indx_to_fill] = seen_item
                        S_grp_scores[group_id][indx_to_fill] = seen_item_score
                    else:
                        min_score_for_grp = np.min(S_grp_scores[group_id][np.isfinite(S_grp_scores[group_id])])
                        if min_score_for_grp < seen_item_score:
                            indx_min = np.argwhere(S_grp_scores[group_id] == min_score_for_grp)[0][
                                0]  # index of min item
                            S_grp_items[group_id][indx_min] = seen_item  # replace min item
                            S_grp_scores[group_id][indx_min] = seen_item_score  # replace min item's score
                # update threshold
                bp_vals = find_bestposition_vals(bp_vals, seen_positions)
                bp_threshold = np.sum(L_scores[bp_vals, np.arange(0, num_lists)])
                row_pos += 1

            indxs_of_finite = np.isfinite(S_grp_scores)
            K_scores = S_grp_scores[indxs_of_finite]
            K_items = S_grp_items[indxs_of_finite]
            total_seen_positions = np.count_nonzero(seen_positions)
            K_items = K_items.astype(int, copy=False)
            return K_items, K_scores, sa_count, ra_count, total_seen_positions, seen_positions

        if np.sum(floors) < k:  # Slack mode
            # Every item's score is less than the threshold and we do not have k elements keep looping
            # use tf check list of lists
            row_pos = 0
            sum_floors = np.sum(floors)
            slack = k - sum_floors
            S_slack_items = np.array([])  # hold slack items
            S_slack_scores = np.array([])  # hold slack items' scores
            while not (np.count_nonzero(S_slack_scores >= bp_threshold) == slack) or not (
                    np.count_nonzero(S_grp_scores >= bp_threshold) == sum_floors):

                if t_style == 'BPA':
                    items_at_pos = L_items[row_pos, :]
                    scores_at_pos = L_scores[row_pos, :]
                    seen_positions[row_pos, :] = 1  # update positions set for sorted access positions
                    sa_count += 1
                else:  # BPA2
                    bp_vals = bp_vals + 1
                    items_at_pos = L_items[bp_vals, np.arange(0, num_lists)]
                    scores_at_pos = L_scores[bp_vals, np.arange(0, num_lists)]
                    if len(np.unique(bp_vals)) == 1:  # SA
                        sa_count += 1
                    else:
                        ra_count += num_lists  # one for each list
                    seen_positions[
                        bp_vals, np.arange(0, num_lists)] = 1  # update positions set for sorted access positions

                unique = np.unique(items_at_pos, return_index=True)
                unique_items = unique[0]
                # random access
                total_score_unique_items = []
                if t_style == "BPA":
                    for i in range(0, unique_items.shape[0]):
                        scores_indxs = np.where(L_items[row_pos + 1:num_items, :] == unique_items[i])
                        updated_rows = scores_indxs[0] + (row_pos + 1)  # update indexes
                        ra_lists = scores_indxs[1]
                        seen_positions[updated_rows, ra_lists] = 1  # update seen positions
                        ra_count += len(ra_lists)
                        total_score_unique_items.append(np.sum(L_scores[updated_rows, scores_indxs[1]]) + np.sum(
                            scores_at_pos[np.where(items_at_pos == unique_items[i])]))
                else:  # BPA2
                    for i in range(0, unique_items.shape[0]):
                        item_to_find = unique_items[i]
                        ra_lists = np.where(items_at_pos != item_to_find)[0]  # lists to search
                        updated_rows = np.asarray([], dtype=int)
                        for list_i in ra_lists:
                            row = np.where(L_items[bp_vals[list_i] + 1:num_items, list_i] == item_to_find)[0][0]
                            updated_rows = np.append(updated_rows, row + bp_vals[list_i] + 1)
                        seen_positions[updated_rows, ra_lists] = 1  # update seen positions
                        ra_count += len(ra_lists)
                        total_score_unique_items.append(np.sum(L_scores[updated_rows, ra_lists]) + np.sum(
                            scores_at_pos[np.where(items_at_pos == item_to_find)]))

                for seen_item_id in range(0, len(unique_items)):
                    seen_item = unique_items[seen_item_id]
                    seen_item_score = total_score_unique_items[seen_item_id]
                    group_id = candidate_db[1, seen_item]
                    grp_floor = floors[np.where(floor_ids == group_id)[0][0]]

                    seen_in_group_bool = seen_item in S_grp_items[group_id]
                    seen_in_S_bool = seen_item in S_slack_items
                    if seen_in_group_bool or seen_in_S_bool:
                        pass
                    # if nothing in S_group or less than floor items add it
                    elif np.count_nonzero(
                            np.isfinite(S_grp_scores[group_id])) < grp_floor and seen_in_group_bool == False:
                        indx_to_fill = np.argwhere(np.isnan(S_grp_scores[group_id]))[0]
                        S_grp_items[group_id][indx_to_fill] = seen_item
                        S_grp_scores[group_id][indx_to_fill] = seen_item_score

                    elif seen_in_group_bool == False:
                        if S_grp_scores[group_id].shape == (0,):  # empty since floor is 0
                            min_score_for_grp = np.Inf
                        else:
                            min_score_for_grp = np.min(S_grp_scores[group_id][np.isfinite(S_grp_scores[group_id])])
                        if min_score_for_grp < seen_item_score:
                            indx_min = np.argwhere(S_grp_scores[group_id] == min_score_for_grp)[0][
                                0]  # index of min item
                            min_item = S_grp_items[group_id][indx_min]  # save min_item
                            S_grp_items[group_id][indx_min] = seen_item  # replace min item
                            S_grp_scores[group_id][indx_min] = seen_item_score  # replace min item's score
                            # Need to update set S with the min item
                            if len(S_slack_scores) < slack:  # if S has less than slack items add it
                                S_slack_items = np.append(S_slack_items, min_item)
                                S_slack_scores = np.append(S_slack_scores, min_score_for_grp)

                            else:
                                min_score = np.min(S_slack_scores)
                                if min_score < min_score_for_grp:  # lowest item in slack set is less than kicked out group item
                                    indx_minS = np.argmin(S_slack_scores)  # index of min item
                                    S_slack_items[indx_minS] = min_item  # replace min item
                                    S_slack_scores[indx_minS] = min_score_for_grp  # replace min item's score



                        # Need to update set S
                        elif len(S_slack_scores) < slack:  # if S has less than k items add it
                            S_slack_scores = np.append(S_slack_scores, seen_item_score)
                            S_slack_items = np.append(S_slack_items, seen_item)

                        elif seen_item_score > np.min(S_slack_scores):  # if item is higher than what is in slack
                            indx_minS = np.argmin(S_slack_scores)  # index of min item
                            S_slack_items[indx_minS] = seen_item  # replace min item
                            S_slack_scores[indx_minS] = seen_item_score  # replace min item's score

                # update threshold
                if t_style == 'BPA':
                    bp_lists = find_bestposition_vals(bp_vals, seen_positions)
                    bp_threshold = np.sum(L_scores[bp_lists, np.arange(0, num_lists)])
                else:  # BPA2
                    bp_vals = find_bestposition_vals(bp_vals, seen_positions)
                    bp_threshold = np.sum(L_scores[bp_vals, np.arange(0, num_lists)])

                row_pos += 1

            indxs_of_finite = np.isfinite(S_grp_scores)
            K_scores = S_grp_scores[indxs_of_finite]
            K_items = S_grp_items[indxs_of_finite]

            K_items = np.hstack((K_items, S_slack_items))
            K_scores = np.hstack((K_scores, S_slack_scores))
            total_seen_positions = np.count_nonzero(seen_positions)
            K_items = K_items.astype(int, copy=False)
            return K_items, K_scores, sa_count, ra_count, total_seen_positions, seen_positions
