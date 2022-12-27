import copy

import numpy as np
import copy
from baselines.aggscores import sorted_aggscores
"""
References: https://yunhefeng.me/material/Gender_Fairness_AAAI2022_Feng.pdf
"""

def episilongreedy(ranking, epsilon, seed):
    current_ranking = copy.deepcopy(ranking)
    np.random.seed(seed) #for reproducibility
    reranking = []
    for i in range(len(current_ranking)):
        p = np.random.rand()
        if p <= epsilon and i < len(current_ranking) - 1: #swap items & can't swap last item
            temp = current_ranking[i]
            j = np.random.randint(i+1, len(current_ranking))
            current_ranking[i] = current_ranking[j]
            current_ranking[j] = temp
            reranking.append(current_ranking[i])
        else: #keep original ranking
            reranking.append(current_ranking[i])

    return np.asarray(reranking)


def baseline_fengetal(epsilon, L_items, L_scores, candidate_db, k, seed):
    """
    Epsilon-Greedy baseline algorithm adapted for fair multi-criteria selection.
    :param epsilon: float in [0,1]
    :param L_items: A numpy array (shape n x m) of the items in the sorted list.
    :param L_scores: A numpy array (shape n x m) of the items' scores in the sorted list.
    :param candidate_db: A numpy array (shape 2 x n) of each item's group membership,
                    first row is item ids, second row is item group membership.
    :param k: Int number of items in subset.
    :param seed: int for reproducibility.
    :return: K_items: A numpy array of items in selected subset, K_scores: A numy array of score of each item in selected subset.
    """

    items_sorted, scores_sorted = sorted_aggscores(L_items, L_scores)
    reranking = episilongreedy(items_sorted, epsilon, seed)

    # reranking_scores = np.asarray(
    #     [scores_sorted[np.argwhere(np.asarray(items_sorted) == item).flatten()[0]] for item in reranking])
    #
    # K_items = reranking[0:k]
    # K_scores = reranking_scores[0:k]
    # reranking_scores = np.asarray([scores_sorted[np.argwhere(np.asarray(items_sorted) == item).flatten()[0]] for item in reranking])
    #
    K_items = reranking[0:k]
    # K_scores = reranking_scores[0:k]
    K_scores = np.asarray(
        [scores_sorted[np.argwhere(np.asarray(items_sorted) == item).flatten()[0]] for item in K_items])

    return K_items, K_scores

