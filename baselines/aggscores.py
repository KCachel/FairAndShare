import numpy as np

def sorted_aggscores(L_items, L_scores):
    """
    Method using sequential sorted access to calculate the aggregate score of every item and then sort the items by descending score.
    :param L_items: A numpy array (shape n x m) of the items in the sorted list.
    :param L_scores: A numpy array (shape n x m) of the items' scores in the sorted list.
    :return: items_sorted: list of item ids sorted by score, scores_sorted: list of scores of items sorted by score.
    """
    num_items, num_lists = np.shape(L_items)
    item_list = []  # hold items as seen
    item_score_list = []  # hold items' scores as seen

    # get items
    # for i_pos in range(0, num_items):
    #     items = L_items[i_pos, :]
    #     item_scores = L_scores[i_pos, :]
    #
    #     for list_i in range(0, num_lists):
    #         seen_item = items[list_i]  # current item
    #         seen_item_score = item_scores[list_i]
    #         if seen_item not in item_list:
    #             item_list.append(seen_item)  # add item to list
    #             item_score_list.append(seen_item_score)  # add first local score
    #         else:
    #             item_indx = item_list.index(seen_item)  # get index of item
    #             item_score_list[item_indx] += seen_item_score  # add score

    for i in range(0, num_items):
        item_list.append(i)
        item_score_list.append(np.sum(L_scores[np.where(L_items == i)]))


    # #sort by descending score
    zipped_item_score_pairs = zip(item_score_list, item_list)

    sorted_item_score_pairs = sorted(zipped_item_score_pairs, reverse=True)

    items_sorted = [item[1] for item in sorted_item_score_pairs]
    scores_sorted = [item[0] for item in sorted_item_score_pairs]
    return items_sorted, scores_sorted