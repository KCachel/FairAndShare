"""Balance Metric
    ----------

"""

# Authors: Anonymous
import numpy as np

def balance(pool_groups, subset_items, subset_groups):
    """Compute the balance error.
    Parameters
    ----------
    pool_groups: numpy array of shape = (n_items)
        The group identity of the items in the pool (corresponding to order of items in pool_items).
    subset_items : numpy array of shape = (n_items)
        The items in the subset(sorted by relevance score).
    subset_groups: numpy array of shape = (n_items)
        The group identity of the items in the subset (corresponding to order of items in subset_items).
    Returns
    ----------
    propOfS: numpy array of shape = (n_groups)
        The proportion of the subset each group is
    bal_val: float
        The balance error
    Examples
    --------
    --------
    >>> pool_items = np.asarray([1,2,3,4])
    >>> pool_scores = np.asarray([100, 85, 54, 12])
    >>> pool_groups = np.asarray([0, 0, 1, 1])
    >>> subset_items = np.asarray([1,4])
    >>> subset_scores = np.asarray([100,12])
    >>> subset_groups = np.asarray([0, 1])
    >>> balance(pool_groups, subset_items, subset_groups)
    [0.5 0.5] 0.0
    """

    #fas.check_subset_items_groups(pool_groups, subset_items, subset_groups) # error handling
    unique_grps = np.unique(pool_groups)
    num_unique_grps = unique_grps.shape[0]
    propOfS  = np.full((num_unique_grps,), -np.Inf)
    total_items_subset = subset_items.shape[0]
    for grp in unique_grps:
        subset_mask = subset_groups == grp
        num_grp_items_in_subset = np.count_nonzero(subset_mask)
        if num_grp_items_in_subset == 0:
            propOfS[grp] = 0
        else:
            propOfS[grp] = num_grp_items_in_subset /total_items_subset

    min_group_proportion_subset  = np.min(propOfS)
    max_group_proportion_subset = np.max(propOfS)
    bal_val =  min_group_proportion_subset / max_group_proportion_subset



    return propOfS, bal_val
