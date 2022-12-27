"""
References: https://github.com/fair-search/fairsearch-fair-pythonana


To use FA*IR: pip install fairsearchcore

"""
import fairsearchcore as fsc
import numpy as np
from fairsearchcore.models import FairScoreDoc
from baselines.aggscores import sorted_aggscores



def baseline_FAIR(pg_FAIR, p, L_items, L_scores, candidate_db, k):
    """
    FA*IR baseline algorithm adapted for fair multi-criteria selection.
    :param pg_FAIR: Int group id of the "protected group".
    :param p: Float "protected group"'s desired representation proportion.
    :param L_items: A numpy array (shape n x m) of the items in the sorted list.
    :param L_scores: A numpy array (shape n x m) of the items' scores in the sorted list.
    :param candidate_db: A numpy array (shape 2 x n) of each item's group membership,
                    first row is item ids, second row is item group membership.
    :param k: Int number of items in subset.
    :return: K_items: A numpy array of items in selected subset, K_scores: A numy array of score of each item in selected subset.
    """


    items_sorted, scores_sorted = sorted_aggscores(L_items, L_scores)



    # k: number of topK elements returned (value should be between 10 and 400)
    _, grp_cnt = np.unique(candidate_db[1,:], return_counts=True)
    print("p value FA*IR: proportion of protected in topk", p)
    alpha = 0.1  # alpha: significance level (value should be between 0.01 and 0.15)

    # create the Fair object
    fair = fsc.Fair(k, p, alpha)

    #create the unfair ranking
    unfair_ranking = [FairScoreDoc(items_sorted[i], scores_sorted[i], candidate_db[1,items_sorted[i]] == pg_FAIR) for i in range(0,len(items_sorted))]




    # now re-rank the unfair ranking
    re_ranked = fair.re_rank(unfair_ranking)
    K_scores = np.asarray([re_ranked[i].score for i in range(0,k)])
    K_items = np.asarray([re_ranked[i].id for i in range(0, k)])

    return K_items, K_scores

def baseline_FAIR_perfcounts(pg_FAIR, p, L_items, L_scores, candidate_db, k):
    """
    FA*IR baseline algorithm adapted for fair multi-criteria selection with performance metrics.
    :param pg_FAIR: Int group id of the "protected group".
    :param p: Float "protected group"'s desired representation proportion.
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


    # k: number of topK elements returned (value should be between 10 and 400)
    _, grp_cnt = np.unique(candidate_db[:,1], return_counts=True)
    print("p value FA*IR: proportion of protected in topk", p)
    alpha = 0.012  # alpha: significance level (value should be between 0.01 and 0.15)

    # create the Fair object
    fair = fsc.Fair(k, p, alpha)

    #create the unfair ranking
    unfair_ranking = [FairScoreDoc(items_sorted[i], scores_sorted[i], candidate_db[items_sorted[i],1] == pg_FAIR) for i in range(0,len(items_sorted))]




    # now re-rank the unfair ranking
    re_ranked = fair.re_rank(unfair_ranking)
    K_scores = np.asarray([re_ranked[i].score for i in range(0,k)])
    K_items = np.asarray([re_ranked[i].id for i in range(0, k)])

    total_seen_positions = num_lists*num_items

    sa_count = num_items
    ra_count = 0

    return K_items, K_scores, sa_count, ra_count, total_seen_positions

