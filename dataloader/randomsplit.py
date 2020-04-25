import numpy as np

import utils

def randomsplit(n_dev, databatch, seed=None):
    """
    :type n_dev: int
    :type databatch: DataBatch
    :type deed: int / None
    :rtype: DataBatch, DataBatch
    """
    n_total = len(databatch)
    assert 0 < n_dev < n_total

    if seed is None:
        indices = np.random.permutation(n_total)
    else:
        indices = np.random.RandomState(seed).permutation(n_total)
    dev_indices = indices[:n_dev]
    train_indices = indices[n_dev:]
    assert len(train_indices) + len(dev_indices) == len(databatch)

    train_databatch = utils.DataBatch(
                            batch_edu_ids=databatch.batch_edu_ids[train_indices],
                            batch_edus=databatch.batch_edus[train_indices],
                            batch_edus_postag=databatch.batch_edus_postag[train_indices],
                            batch_edus_head=databatch.batch_edus_head[train_indices],
                            batch_sbnds=databatch.batch_sbnds[train_indices],
                            batch_pbnds=databatch.batch_pbnds[train_indices],
                            batch_nary_sexp=databatch.batch_nary_sexp[train_indices],
                            batch_bin_sexp=databatch.batch_bin_sexp[train_indices],
                            batch_arcs=databatch.batch_arcs[train_indices])
    dev_databatch = utils.DataBatch(
                            batch_edu_ids=databatch.batch_edu_ids[dev_indices],
                            batch_edus=databatch.batch_edus[dev_indices],
                            batch_edus_postag=databatch.batch_edus_postag[dev_indices],
                            batch_edus_head=databatch.batch_edus_head[dev_indices],
                            batch_sbnds=databatch.batch_sbnds[dev_indices],
                            batch_pbnds=databatch.batch_pbnds[dev_indices],
                            batch_nary_sexp=databatch.batch_nary_sexp[dev_indices],
                            batch_bin_sexp=databatch.batch_bin_sexp[dev_indices],
                            batch_arcs=databatch.batch_arcs[dev_indices])

    utils.writelog("n_dev=%d" % n_dev)
    utils.writelog("# of training instances=%d" % len(train_databatch))
    utils.writelog("# of development instances=%d" % len(dev_databatch))

    return train_databatch, dev_databatch

