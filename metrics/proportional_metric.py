"""Statistical Parity Metric
    ----------

"""

# Authors: Anonymous


import numpy as np



def parity(pool_items, pool_groups, subset_items, subset_groups):
    """Compute the error in selection statistical parity.
    Parameters
    ----------
    pool_items : numpy array of shape = (n_items)
        The items in the pool (sorted by relevance score).
    pool_groups: numpy array of shape = (n_items)
        The group identity of the items in the pool (corresponding to order of items in pool_items).
    subset_items : numpy array of shape = (n_items)
        The items in the subset(sorted by relevance score).
    subset_groups: numpy array of shape = (n_items)
        The group identity of the items in the subset (corresponding to order of items in subset_items).
    Returns
    ----------
    selectRt: numpy array of shape = (n_groups)
        The proportion of each group selected into the subset
    sp_val: float
        Parity fairness.
    Examples
    --------
    --------
    >>> pool_items = np.asarray([1,2,3,4])
    >>> pool_groups = np.asarray([0, 0, 1, 1])
    >>> subset_items = np.asarray([1,4])
    >>> subset_groups = np.asarray([0, 1])
    >>> parity(pool_items, pool_groups, subset_items, subset_groups)
    [0.5 0.5] 1.0
    """

    #fas.check_pool_subset_items_groups(pool_items, pool_groups, subset_items, subset_groups) #error handling
    unique_grps = np.unique(pool_groups)
    num_unique_grps = unique_grps.shape[0]
    selectRt  = np.full((num_unique_grps,), -np.Inf)
    for grp in unique_grps:
        pool_mask = pool_groups == grp
        subset_mask = subset_groups == grp
        num_grp_items_in_subset = np.count_nonzero(subset_mask)
        num_grp_items_in_pool = np.count_nonzero(pool_mask)
        if num_grp_items_in_subset == 0:
            selectRt[grp] = 0
        else:
            selectRt[grp] = num_grp_items_in_subset /num_grp_items_in_pool

    min_group_selection_prop  = np.min(selectRt)
    max_group_selection_prop = np.max(selectRt)
    sp_val = min_group_selection_prop / max_group_selection_prop
    return selectRt, sp_val
