import numpy as np
from sklearn.model_selection import KFold

import utils

def kfold(n_splits, split_id, databatch):
    """
    :type n_splits: int
    :type split_id: int
    :type databatch: DataBatch
    :rtype: DataBatch, DataBatch
    """
    assert 0 <= split_id < n_splits

    kfold = KFold(n_splits=n_splits, random_state=1234, shuffle=True)

    indices_list = list(kfold.split(np.arange(len(databatch))))
    train_indices, dev_indices = indices_list[split_id]
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

    utils.writelog("n_splits=%d" % n_splits)
    utils.writelog("split_id=%d" % split_id)
    utils.writelog("# of training instances=%d" % len(train_databatch))
    utils.writelog("# of development instances=%d" % len(dev_databatch))

    return train_databatch, dev_databatch


