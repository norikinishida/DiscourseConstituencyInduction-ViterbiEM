import jsonlines
import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, optimizers, serializers
import multiset
import pyprind

import utils
import treetk

import treesamplers
import parsing
import metrics

# def train(model,
#           decoder,
#           sampler,
#           max_epoch,
#           n_init_epochs,
#           negative_size,
#           batch_size,
#           weight_decay,
#           gradient_clipping,
#           optimizer_name,
#           train_dataset,
#           dev_dataset,
#           path_train,
#           path_valid,
#           path_snapshot,
#           path_pred,
#           path_gold):
#     """
#     :type model: Model
#     :type decoder: IncrementalCKYDecoder
#     :type sampler: TreeSampler
#     :type max_epoch: int
#     :type n_init_epochs: int
#     :type negative_size: int
#     :type batch_size: int
#     :type weight_decay: float
#     :type gradient_clipping: float
#     :type optimizer_name: str
#     :type train_dataset: numpy.ndarray
#     :type dev_dataset: numpy.ndarray
#     :type path_train: str
#     :type path_valid: str
#     :type path_snapshot: str
#     :type path_pred: str
#     :type path_gold: str
#     :rtype: None
#     """
#     writer_train = jsonlines.Writer(open(path_train, "w"), flush=True)
#     if dev_dataset is not None:
#         writer_valid = jsonlines.Writer(open(path_valid, "w"), flush=True)
#
#     boundary_flags = [(True,False)]
#     assert negative_size >= len(boundary_flags)
#     negative_tree_sampler = treesamplers.NegativeTreeSampler()
#
#     # Optimizer preparation
#     if optimizer_name == "adam":
#         opt = optimizers.Adam()
#     else:
#         raise ValueError("Invalid optimizer_name=%s" % optimizer_name)
#
#     opt.setup(model)
#
#     if weight_decay > 0.0:
#         opt.add_hook(chainer.optimizer.WeightDecay(weight_decay))
#     if gradient_clipping:
#         opt.add_hook(chainer.optimizer.GradientClipping(gradient_clipping))
#
#     n_train = len(train_dataset)
#     it = 0
#     bestscore_holder = utils.BestScoreHolder(scale=100.0)
#     bestscore_holder.init()
#
#     if dev_dataset is not None:
#         # Initial validation
#         with chainer.using_config("train", False), chainer.no_backprop_mode():
#             parsing.parse(model=model,
#                           decoder=decoder,
#                           dataset=dev_dataset,
#                           path_pred=path_pred)
#             scores = metrics.rst_parseval(
#                         pred_path=path_pred,
#                         gold_path=path_gold)
#             old_scores = metrics.old_rst_parseval(
#                         pred_path=path_pred,
#                         gold_path=path_gold)
#             out = {
#                     "epoch": 0,
#                     "Morey2018": {
#                         "Unlabeled Precision": scores["S"]["Precision"] * 100.0,
#                         "Precision_info": scores["S"]["Precision_info"],
#                         "Unlabeled Recall": scores["S"]["Recall"] * 100.0,
#                         "Recall_info": scores["S"]["Recall_info"],
#                         "Micro F1": scores["S"]["Micro F1"] * 100.0},
#                     "Marcu2000": {
#                         "Unlabeled Precision": old_scores["S"]["Precision"] * 100.0,
#                         "Precision_info": old_scores["S"]["Precision_info"],
#                         "Unlabeled Recall": old_scores["S"]["Recall"] * 100.0,
#                         "Recall_info": old_scores["S"]["Recall_info"],
#                         "Micro F1": old_scores["S"]["Micro F1"] * 100.0}}
#             writer_valid.write(out)
#             utils.writelog(utils.pretty_format_dict(out))
#         # Saving
#         bestscore_holder.compare_scores(scores["S"]["Micro F1"], step=0)
#         serializers.save_npz(path_snapshot, model)
#         utils.writelog("Saved the model to %s" % path_snapshot)
#     else:
#         # Saving
#         serializers.save_npz(path_snapshot, model)
#         utils.writelog("Saved the model to %s" % path_snapshot)
#
#     for epoch in range(1, max_epoch+1):
#
#         perm = np.random.permutation(n_train)
#
#         for inst_i in range(0, n_train, batch_size):
#
#             ### Mini batch
#
#             # Init
#             loss_constituency, acc_constituency = 0.0, 0.0
#             actual_batchsize = 0
#
#             for data in train_dataset[perm[inst_i:inst_i+batch_size]]:
#
#                 ### One data instance
#
#                 edu_ids = data.edu_ids
#                 edus = data.edus
#                 edus_postag = data.edus_postag
#                 edus_head = data.edus_head
#                 sbnds = data.sbnds
#                 pbnds = data.pbnds
#
#                 # Feature extraction
#                 edu_vectors = model.forward_edus(edus, edus_postag, edus_head) # (n_edus, bilstm_dim)
#                 padded_edu_vectors = model.pad_edu_vectors(edu_vectors) # (n_edus+2, bilstm_dim)
#                 mask_bwd, mask_fwd = model.make_masks() # (1, bilstm_dim), (1, bilstm_dim)
#
#                 ########## E-Step (BEGIN) ##########
#                 # Positive tree
#                 with chainer.using_config("train", False), chainer.no_backprop_mode():
#
#                     if epoch <= n_init_epochs:
#                         pos_sexp = sampler.sample(
#                                         sexps=edu_ids,
#                                         edus=edus,
#                                         edus_head=edus_head,
#                                         sbnds=sbnds,
#                                         pbnds=pbnds)
#                     else:
#                         pos_sexp = decoder.decode(
#                                         model=model,
#                                         sexps=edu_ids,
#                                         edus=edus,
#                                         edus_postag=edus_postag,
#                                         sbnds=sbnds,
#                                         pbnds=pbnds,
#                                         padded_edu_vectors=padded_edu_vectors,
#                                         mask_bwd=mask_bwd,
#                                         mask_fwd=mask_fwd,
#                                         use_sbnds=True,
#                                         use_pbnds=True)
#                     pos_tree = treetk.sexp2tree(pos_sexp, with_nonterminal_labels=False, with_terminal_labels=False)
#                     pos_tree.calc_spans()
#                     pos_spans = treetk.aggregate_spans(pos_tree, include_terminal=False, order="post-order") # list of (int, int)
#                 ########## E-Step (END) ##########
#
#                 ########## M-Step-1 (BEGIN) ##########
#                 # Negative tree
#                 pos_neg_spans = []
#                 margins = []
#                 pos_neg_spans.append(pos_spans)
#                 with chainer.using_config("train", False), chainer.no_backprop_mode():
#                     for use_sbnds,use_pbnds in boundary_flags:
#                         neg_bin_sexp = decoder.decode(
#                                             model=model,
#                                             sexps=edu_ids,
#                                             edus=edus,
#                                             edus_postag=edus_postag,
#                                             sbnds=sbnds,
#                                             pbnds=pbnds,
#                                             padded_edu_vectors=padded_edu_vectors,
#                                             mask_bwd=mask_bwd,
#                                             mask_fwd=mask_fwd,
#                                             use_sbnds=use_sbnds,
#                                             use_pbnds=use_pbnds,
#                                             gold_spans=pos_spans) # list of str
#                         neg_tree = treetk.sexp2tree(neg_bin_sexp, with_nonterminal_labels=False, with_terminal_labels=False)
#                         neg_tree.calc_spans()
#                         neg_spans = treetk.aggregate_spans(neg_tree, include_terminal=False, order="pre-order") # list of (int, int)
#                         margin = compute_tree_distance(pos_spans, neg_spans, coef=1.0)
#                         pos_neg_spans.append(neg_spans)
#                         margins.append(margin)
#                 for _ in range(negative_size - len(boundary_flags)):
#                     neg_bin_sexp = negative_tree_sampler.sample(sexps=edu_ids, sbnds=sbnds, pbnds=pbnds)
#                     neg_tree = treetk.sexp2tree(neg_bin_sexp, with_nonterminal_labels=False, with_terminal_labels=False)
#                     neg_tree.calc_spans()
#                     neg_spans = treetk.aggregate_spans(neg_tree, include_terminal=False, order="pre-order") # list of (int, int)
#                     margin = compute_tree_distance(pos_spans, neg_spans, coef=1.0)
#                     pos_neg_spans.append(neg_spans)
#                     margins.append(margin)
#
#                 # Scoring
#                 pred_scores = model.forward_spans_for_bracketing(
#                                         edus=edus,
#                                         edus_postag=edus_postag,
#                                         sbnds=sbnds,
#                                         pbnds=pbnds,
#                                         padded_edu_vectors=padded_edu_vectors,
#                                         mask_bwd=mask_bwd,
#                                         mask_fwd=mask_fwd,
#                                         batch_spans=pos_neg_spans,
#                                         aggregate=True) # (1+negative_size, 1)
#
#                 # Constituency Loss
#                 for neg_i in range(negative_size):
#                     loss_constituency += F.clip(pred_scores[1+neg_i] + margins[neg_i] - pred_scores[0], 0.0, 10000000.0)
#                 ########## M-Step-1 (END) ##########
#
#                 # Ranked Accuracy
#                 pred_scores = F.reshape(pred_scores, (1, 1+negative_size)) # (1, 1+negative_size)
#                 gold_scores = np.zeros((1,), dtype=np.int32) # (1,)
#                 gold_scores = utils.convert_ndarray_to_variable(gold_scores, seq=False) # (1,)
#                 acc_constituency += F.accuracy(pred_scores, gold_scores)
#
#                 actual_batchsize += 1
#
#             ########## M-Step-2 (BEGIN) ##########
#             # Backward & Update
#             actual_batchsize = float(actual_batchsize)
#             loss_constituency = loss_constituency / actual_batchsize
#             acc_constituency = acc_constituency / actual_batchsize
#             loss = loss_constituency
#             model.zerograds()
#             loss.backward()
#             opt.update()
#             it += 1
#             ########## M-Step-2 (END) ##########
#
#             # Write log
#             loss_constituency_data = float(cuda.to_cpu(loss_constituency.data))
#             acc_constituency_data = float(cuda.to_cpu(acc_constituency.data))
#             out = {"iter": it,
#                    "epoch": epoch,
#                    "progress": "%d/%d" % (inst_i+actual_batchsize, n_train),
#                    "progress_ratio": float(inst_i+actual_batchsize)/n_train*100.0,
#                    "Constituency Loss": loss_constituency_data,
#                    "Ranked Accuracy": acc_constituency_data * 100.0}
#             writer_train.write(out)
#             utils.writelog(utils.pretty_format_dict(out))
#             print(bestscore_holder.best_score * 100.0)
#
#         if dev_dataset is not None:
#             # Validation
#             with chainer.using_config("train", False), chainer.no_backprop_mode():
#                 parsing.parse(model=model,
#                               decoder=decoder,
#                               dataset=dev_dataset,
#                               path_pred=path_pred)
#                 scores = metrics.rst_parseval(
#                             pred_path=path_pred,
#                             gold_path=path_gold)
#                 old_scores = metrics.old_rst_parseval(
#                             pred_path=path_pred,
#                             gold_path=path_gold)
#                 out = {
#                         "epoch": epoch,
#                         "Morey2018": {
#                             "Unlabeled Precision": scores["S"]["Precision"] * 100.0,
#                             "Precision_info": scores["S"]["Precision_info"],
#                             "Unlabeled Recall": scores["S"]["Recall"] * 100.0,
#                             "Recall_info": scores["S"]["Recall_info"],
#                             "Micro F1": scores["S"]["Micro F1"] * 100.0},
#                         "Marcu2000": {
#                             "Unlabeled Precision": old_scores["S"]["Precision"] * 100.0,
#                             "Precision_info": old_scores["S"]["Precision_info"],
#                             "Unlabeled Recall": old_scores["S"]["Recall"] * 100.0,
#                             "Recall_info": old_scores["S"]["Recall_info"],
#                             "Micro F1": old_scores["S"]["Micro F1"] * 100.0}}
#                 writer_valid.write(out)
#                 utils.writelog(utils.pretty_format_dict(out))
#             # Saving
#             did_update = bestscore_holder.compare_scores(scores["S"]["Micro F1"], epoch)
#             if did_update:
#                 serializers.save_npz(path_snapshot, model)
#                 utils.writelog("Saved the model to %s" % path_snapshot)
#             # Finished?
#             if bestscore_holder.ask_finishing(max_patience=10):
#                 utils.writelog("Patience %d is over. Training finished successfully." % bestscore_holder.patience)
#                 writer_train.close()
#                 if dev_dataset is not None:
#                     writer_valid.close()
#                 return
#         else:
#             # No validation
#             # Saving
#             serializers.save_npz(path_snapshot, model)
#             # We continue training until it reaches the maximum number of epochs.

def train(model,
          decoder,
          sampler,
          max_epoch,
          n_init_epochs,
          negative_size,
          batch_size,
          weight_decay,
          gradient_clipping,
          optimizer_name,
          train_dataset,
          dev_dataset,
          path_train,
          path_valid,
          path_snapshot,
          path_pred,
          path_gold):
    """
    :type model: Model
    :type decoder: IncrementalCKYDecoder
    :type sampler: TreeSampler
    :type max_epoch: int
    :type n_init_epochs: int
    :type negative_size: int
    :type batch_size: int
    :type weight_decay: float
    :type gradient_clipping: float
    :type optimizer_name: str
    :type train_dataset: numpy.ndarray
    :type dev_dataset: numpy.ndarray
    :type path_train: str
    :type path_valid: str
    :type path_snapshot: str
    :type path_pred: str
    :type path_gold: str
    :rtype: None
    """
    writer_train = jsonlines.Writer(open(path_train, "w"), flush=True)
    if dev_dataset is not None:
        writer_valid = jsonlines.Writer(open(path_valid, "w"), flush=True)

    boundary_flags = [(True,False)]
    assert negative_size >= len(boundary_flags)
    negative_tree_sampler = treesamplers.NegativeTreeSampler()

    # Optimizer preparation
    if optimizer_name == "adam":
        opt = optimizers.Adam()
    else:
        raise ValueError("Invalid optimizer_name=%s" % optimizer_name)

    opt.setup(model)

    if weight_decay > 0.0:
        opt.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    if gradient_clipping:
        opt.add_hook(chainer.optimizer.GradientClipping(gradient_clipping))

    n_train = len(train_dataset)
    it = 0
    bestscore_holder = utils.BestScoreHolder(scale=100.0)
    bestscore_holder.init()

    if dev_dataset is not None:
        # Initial validation
        with chainer.using_config("train", False), chainer.no_backprop_mode():
            parsing.parse(model=model,
                          decoder=decoder,
                          dataset=dev_dataset,
                          path_pred=path_pred)
            scores = metrics.rst_parseval(
                        pred_path=path_pred,
                        gold_path=path_gold)
            old_scores = metrics.old_rst_parseval(
                        pred_path=path_pred,
                        gold_path=path_gold)
            out = {
                    "epoch": 0,
                    "Morey2018": {
                        "Unlabeled Precision": scores["S"]["Precision"] * 100.0,
                        "Precision_info": scores["S"]["Precision_info"],
                        "Unlabeled Recall": scores["S"]["Recall"] * 100.0,
                        "Recall_info": scores["S"]["Recall_info"],
                        "Micro F1": scores["S"]["Micro F1"] * 100.0},
                    "Marcu2000": {
                        "Unlabeled Precision": old_scores["S"]["Precision"] * 100.0,
                        "Precision_info": old_scores["S"]["Precision_info"],
                        "Unlabeled Recall": old_scores["S"]["Recall"] * 100.0,
                        "Recall_info": old_scores["S"]["Recall_info"],
                        "Micro F1": old_scores["S"]["Micro F1"] * 100.0}}
            writer_valid.write(out)
            utils.writelog(utils.pretty_format_dict(out))
        # Saving
        bestscore_holder.compare_scores(scores["S"]["Micro F1"], step=0)
        serializers.save_npz(path_snapshot, model)
        utils.writelog("Saved the model to %s" % path_snapshot)
    else:
        # Saving
        serializers.save_npz(path_snapshot, model)
        utils.writelog("Saved the model to %s" % path_snapshot)

    for epoch in range(1, max_epoch+1):

        perm = np.random.permutation(n_train)

        ########## E-Step (BEGIN) ##########
        utils.writelog("E step ===>")

        prog_bar = pyprind.ProgBar(n_train)

        for inst_i in range(0, n_train, batch_size):

            ### Mini batch

            for data in train_dataset[inst_i:inst_i+batch_size]:

                ### One data instance

                edu_ids = data.edu_ids
                edus = data.edus
                edus_postag = data.edus_postag
                edus_head = data.edus_head
                sbnds = data.sbnds
                pbnds = data.pbnds

                with chainer.using_config("train", False), chainer.no_backprop_mode():

                    # Feature extraction
                    edu_vectors = model.forward_edus(edus, edus_postag, edus_head) # (n_edus, bilstm_dim)
                    padded_edu_vectors = model.pad_edu_vectors(edu_vectors) # (n_edus+2, bilstm_dim)
                    mask_bwd, mask_fwd = model.make_masks() # (1, bilstm_dim), (1, bilstm_dim)

                    # Positive tree
                    if epoch <= n_init_epochs:
                        pos_sexp = sampler.sample(
                                        sexps=edu_ids,
                                        edus=edus,
                                        edus_head=edus_head,
                                        sbnds=sbnds,
                                        pbnds=pbnds)
                    else:
                        pos_sexp = decoder.decode(
                                        model=model,
                                        sexps=edu_ids,
                                        edus=edus,
                                        edus_postag=edus_postag,
                                        sbnds=sbnds,
                                        pbnds=pbnds,
                                        padded_edu_vectors=padded_edu_vectors,
                                        mask_bwd=mask_bwd,
                                        mask_fwd=mask_fwd,
                                        use_sbnds=True,
                                        use_pbnds=True)
                    pos_tree = treetk.sexp2tree(pos_sexp, with_nonterminal_labels=False, with_terminal_labels=False)
                    pos_tree.calc_spans()
                    pos_spans = treetk.aggregate_spans(pos_tree, include_terminal=False, order="post-order") # list of (int, int)
                    data.pos_spans = pos_spans  #NOTE
                    prog_bar.update()
        ########## E-Step (END) ##########

        ########## M-Step (BEGIN) ##########
        utils.writelog("M step ===>")

        for inst_i in range(0, n_train, batch_size):

            # Processing one mini-batch

            # Init
            loss_constituency, acc_constituency = 0.0, 0.0
            actual_batchsize = 0

            for data in train_dataset[perm[inst_i:inst_i+batch_size]]:

                # Processing one instance

                edu_ids = data.edu_ids
                edus = data.edus
                edus_postag = data.edus_postag
                edus_head = data.edus_head
                sbnds = data.sbnds
                pbnds = data.pbnds
                pos_spans = data.pos_spans # NOTE

                # Feature extraction
                edu_vectors = model.forward_edus(edus, edus_postag, edus_head) # (n_edus, bilstm_dim)
                padded_edu_vectors = model.pad_edu_vectors(edu_vectors) # (n_edus+2, bilstm_dim)
                mask_bwd, mask_fwd = model.make_masks() # (1, bilstm_dim), (1, bilstm_dim)

                # Negative trees
                pos_neg_spans = []
                margins = []
                pos_neg_spans.append(pos_spans)
                with chainer.using_config("train", False), chainer.no_backprop_mode():
                    for use_sbnds, use_pbnds in boundary_flags:
                        neg_bin_sexp = decoder.decode(
                                            model=model,
                                            sexps=edu_ids,
                                            edus=edus,
                                            edus_postag=edus_postag,
                                            sbnds=sbnds,
                                            pbnds=pbnds,
                                            padded_edu_vectors=padded_edu_vectors,
                                            mask_bwd=mask_bwd,
                                            mask_fwd=mask_fwd,
                                            use_sbnds=use_sbnds,
                                            use_pbnds=use_pbnds,
                                            gold_spans=pos_spans) # list of str
                        neg_tree = treetk.sexp2tree(neg_bin_sexp, with_nonterminal_labels=False, with_terminal_labels=False)
                        neg_tree.calc_spans()
                        neg_spans = treetk.aggregate_spans(neg_tree, include_terminal=False, order="pre-order") # list of (int, int)
                        margin = compute_tree_distance(pos_spans, neg_spans, coef=1.0)
                        pos_neg_spans.append(neg_spans)
                        margins.append(margin)
                for _ in range(negative_size - len(boundary_flags)):
                    neg_bin_sexp = negative_tree_sampler.sample(sexps=edu_ids, sbnds=sbnds, pbnds=pbnds)
                    neg_tree = treetk.sexp2tree(neg_bin_sexp, with_nonterminal_labels=False, with_terminal_labels=False)
                    neg_tree.calc_spans()
                    neg_spans = treetk.aggregate_spans(neg_tree, include_terminal=False, order="pre-order") # list of (int, int)
                    margin = compute_tree_distance(pos_spans, neg_spans, coef=1.0)
                    pos_neg_spans.append(neg_spans)
                    margins.append(margin)

                # Scoring
                pred_scores = model.forward_spans_for_bracketing(
                                        edus=edus,
                                        edus_postag=edus_postag,
                                        sbnds=sbnds,
                                        pbnds=pbnds,
                                        padded_edu_vectors=padded_edu_vectors,
                                        mask_bwd=mask_bwd,
                                        mask_fwd=mask_fwd,
                                        batch_spans=pos_neg_spans,
                                        aggregate=True) # (1+negative_size, 1)

                # Constituency Loss
                for neg_i in range(negative_size):
                    loss_constituency += F.clip(pred_scores[1+neg_i] + margins[neg_i] - pred_scores[0], 0.0, 10000000.0)

                # Ranked Accuracy
                pred_scores = F.reshape(pred_scores, (1, 1+negative_size)) # (1, 1+negative_size)
                gold_scores = np.zeros((1,), dtype=np.int32) # (1,)
                gold_scores = utils.convert_ndarray_to_variable(gold_scores, seq=False) # (1,)
                acc_constituency += F.accuracy(pred_scores, gold_scores)

                actual_batchsize += 1

            # Backward & Update
            actual_batchsize = float(actual_batchsize)
            loss_constituency = loss_constituency / actual_batchsize
            acc_constituency = acc_constituency / actual_batchsize
            loss = loss_constituency
            model.zerograds()
            loss.backward()
            opt.update()
            it += 1

            # Write log
            loss_constituency_data = float(cuda.to_cpu(loss_constituency.data))
            acc_constituency_data = float(cuda.to_cpu(acc_constituency.data))
            out = {"iter": it,
                   "epoch": epoch,
                   "progress": "%d/%d" % (inst_i+actual_batchsize, n_train),
                   "progress_ratio": float(inst_i+actual_batchsize)/n_train*100.0,
                   "Constituency Loss": loss_constituency_data,
                   "Ranked Accuracy": acc_constituency_data * 100.0}
            writer_train.write(out)
            utils.writelog(utils.pretty_format_dict(out))
            print(bestscore_holder.best_score * 100.0)
        ########## M-Step (END) ##########

        if dev_dataset is not None:
            # Validation
            with chainer.using_config("train", False), chainer.no_backprop_mode():
                parsing.parse(model=model,
                              decoder=decoder,
                              dataset=dev_dataset,
                              path_pred=path_pred)
                scores = metrics.rst_parseval(
                            pred_path=path_pred,
                            gold_path=path_gold)
                old_scores = metrics.old_rst_parseval(
                            pred_path=path_pred,
                            gold_path=path_gold)
                out = {
                        "epoch": epoch,
                        "Morey2018": {
                            "Unlabeled Precision": scores["S"]["Precision"] * 100.0,
                            "Precision_info": scores["S"]["Precision_info"],
                            "Unlabeled Recall": scores["S"]["Recall"] * 100.0,
                            "Recall_info": scores["S"]["Recall_info"],
                            "Micro F1": scores["S"]["Micro F1"] * 100.0},
                        "Marcu2000": {
                            "Unlabeled Precision": old_scores["S"]["Precision"] * 100.0,
                            "Precision_info": old_scores["S"]["Precision_info"],
                            "Unlabeled Recall": old_scores["S"]["Recall"] * 100.0,
                            "Recall_info": old_scores["S"]["Recall_info"],
                            "Micro F1": old_scores["S"]["Micro F1"] * 100.0}}
                writer_valid.write(out)
                utils.writelog(utils.pretty_format_dict(out))
            # Saving
            did_update = bestscore_holder.compare_scores(scores["S"]["Micro F1"], epoch)
            if did_update:
                serializers.save_npz(path_snapshot, model)
                utils.writelog("Saved the model to %s" % path_snapshot)
            # Finished?
            if bestscore_holder.ask_finishing(max_patience=10):
                utils.writelog("Patience %d is over. Training finished successfully." % bestscore_holder.patience)
                writer_train.close()
                if dev_dataset is not None:
                    writer_valid.close()
                return
        else:
            # No validation
            # Saving
            serializers.save_npz(path_snapshot, model)
            # We continue training until it reaches the maximum number of epochs.

def compute_tree_distance(spans1, spans2, coef):
    """
    :type spans1: list of (int, int)
    :type spans2: list of (int, int)
    :type coef: float
    :rtype: float
    """
    assert len(spans1) == len(spans2)

    spans1 = multiset.Multiset(spans1)
    spans2 = multiset.Multiset(spans2)

    assert len(spans1) == len(spans2)
    dist = len(spans1) - len(spans1 & spans2)
    dist = float(dist)

    dist = coef * dist
    return dist

