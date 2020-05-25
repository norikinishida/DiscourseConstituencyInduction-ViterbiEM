import argparse
import os
import random
import time

import jsonlines
import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, optimizers, serializers
import multiset
import pyprind

import utils
import treetk

import dataloader
import models
import treesamplers
import decoders
import metrics

def main(args):

    ####################
    # Arguments
    gpu = args.gpu
    model_name = args.model
    initial_tree_sampling = args.initial_tree_sampling
    path_config = args.config
    data_augmentation = args.data_augmentation
    trial_name = args.name
    actiontype = args.actiontype
    max_epoch = args.max_epoch
    dev_size = args.dev_size

    # Check
    assert actiontype in ["train", "evaluate"]
    if actiontype == "train":
        assert max_epoch > 0
    assert len(initial_tree_sampling.split("_")) == 3
    for type_ in initial_tree_sampling.split("_"):
        assert type_ in ["X", "BU", "TD", "RB", "LB", "RB2"]
    assert initial_tree_sampling.split("_")[2] != "X"
    assert initial_tree_sampling.split("_")[1] != "RB2"
    assert initial_tree_sampling.split("_")[2] != "RB2"

    if trial_name is None or trial_name == "None":
        trial_name = utils.get_current_time()

    ####################
    # Path setting
    config = utils.Config(path_config)

    basename = "%s.%s.%s.aug_%s.%s" \
            % (model_name,
               initial_tree_sampling,
               utils.get_basename_without_ext(path_config),
               data_augmentation,
               trial_name)

    if actiontype == "train":
        path_log = os.path.join(config.getpath("results"), basename + ".training.log")
    elif actiontype == "evaluate":
        path_log = os.path.join(config.getpath("results"), basename + ".evaluation.log")
    path_train = os.path.join(config.getpath("results"), basename + ".training.jsonl")
    path_valid = os.path.join(config.getpath("results"), basename + ".validation.jsonl")
    path_snapshot = os.path.join(config.getpath("results"), basename + ".model")
    path_pred = os.path.join(config.getpath("results"), basename + ".evaluation.ctrees")
    path_eval = os.path.join(config.getpath("results"), basename + ".evaluation.json")

    utils.set_logger(path_log)

    ####################
    # Random seed
    random_seed = trial_name
    random_seed = utils.hash_string(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    cuda.cupy.random.seed(random_seed)

    ####################
    # Log so far
    utils.writelog("gpu=%d" % gpu)
    utils.writelog("model_name=%s" % model_name)
    utils.writelog("initial_tree_sampling=%s" % initial_tree_sampling)
    utils.writelog("path_config=%s" % path_config)
    utils.writelog("data_augmentation=%s" % data_augmentation)
    utils.writelog("trial_name=%s" % trial_name)
    utils.writelog("actiontype=%s" % actiontype)
    utils.writelog("max_epoch=%s" % max_epoch)
    utils.writelog("dev_size=%s" % dev_size)

    utils.writelog("path_log=%s" % path_log)
    utils.writelog("path_train=%s" % path_train)
    utils.writelog("path_valid=%s" % path_valid)
    utils.writelog("path_snapshot=%s" % path_snapshot)
    utils.writelog("path_pred=%s" % path_pred)
    utils.writelog("path_eval=%s" % path_eval)

    utils.writelog("random_seed=%d" % random_seed)

    ####################
    # Data preparation
    begin_time = time.time()

    train_dataset = dataloader.read_rstdt("train", relation_level="coarse-grained", with_root=False)
    test_dataset = dataloader.read_rstdt("test", relation_level="coarse-grained", with_root=False)
    vocab_word = utils.read_vocab(os.path.join(config.getpath("data"), "rstdt-vocab", "words.vocab.txt"))
    vocab_postag = utils.read_vocab(os.path.join(config.getpath("data"), "rstdt-vocab", "postags.vocab.txt"))
    vocab_deprel = utils.read_vocab(os.path.join(config.getpath("data"), "rstdt-vocab", "deprels.vocab.txt"))

    if data_augmentation:
        external_train_dataset= dataloader.read_ptbwsj_wo_rstdt(with_root=False)
        # Remove documents with only one leaf node
        external_train_dataset = utils.filter_dataset(external_train_dataset, condition=lambda data: len(data.edu_ids) > 1)

    end_time = time.time()
    utils.writelog("Loaded the corpus. %f [sec.]" % (end_time - begin_time))

    ####################
    # Hyper parameters
    word_dim = config.getint("word_dim")
    postag_dim = config.getint("postag_dim")
    deprel_dim = config.getint("deprel_dim")
    lstm_dim = config.getint("lstm_dim")
    mlp_dim = config.getint("mlp_dim")
    n_init_epochs = config.getint("n_init_epochs")
    negative_size = config.getint("negative_size")
    batch_size = config.getint("batch_size")
    weight_decay = config.getfloat("weight_decay")
    gradient_clipping = config.getfloat("gradient_clipping")
    optimizer_name = config.getstr("optimizer_name")

    utils.writelog("word_dim=%d" % word_dim)
    utils.writelog("postag_dim=%d" % postag_dim)
    utils.writelog("deprel_dim=%d" % deprel_dim)
    utils.writelog("lstm_dim=%d" % lstm_dim)
    utils.writelog("mlp_dim=%d" % mlp_dim)
    utils.writelog("n_init_epochs=%d" % n_init_epochs)
    utils.writelog("negative_size=%d" % negative_size)
    utils.writelog("batch_size=%d" % batch_size)
    utils.writelog("weight_decay=%f" % weight_decay)
    utils.writelog("gradient_clipping=%f" % gradient_clipping)
    utils.writelog("optimizer_name=%s" % optimizer_name)

    ####################
    # Model preparation
    cuda.get_device(gpu).use()

    # Initialize a model
    utils.mkdir(os.path.join(config.getpath("data"), "caches"))
    path_embed = config.getpath("pretrained_word_embeddings")
    path_caches = os.path.join(config.getpath("data"), "caches", "cached." + os.path.basename(path_embed) + ".npy")
    if os.path.exists(path_caches):
        utils.writelog("Loading cached word embeddings ...")
        initialW = np.load(path_caches)
    else:
        initialW = utils.read_word_embedding_matrix(
                                path=path_embed,
                                dim=word_dim,
                                vocab=vocab_word,
                                scale=0.0)
        np.save(path_caches, initialW)

    if model_name == "spanbasedmodel":
        # Span-based model w/ template features
        template_feature_extractor = models.TemplateFeatureExtractor(
                                                dataset=train_dataset)
        utils.writelog("Template feature size=%d" % template_feature_extractor.feature_size)
        if actiontype == "train":
            for template in template_feature_extractor.templates:
                dim = template_feature_extractor.template2dim[template]
                utils.writelog("Template feature #%s %s" % (dim, template))
        model = models.SpanBasedModel(
                        vocab_word=vocab_word,
                        vocab_postag=vocab_postag,
                        vocab_deprel=vocab_deprel,
                        word_dim=word_dim,
                        postag_dim=postag_dim,
                        deprel_dim=deprel_dim,
                        lstm_dim=lstm_dim,
                        mlp_dim=mlp_dim,
                        initialW=initialW,
                        template_feature_extractor=template_feature_extractor)
    elif model_name == "spanbasedmodel2":
        # Span-based model w/o template features
        model = models.SpanBasedModel2(
                        vocab_word=vocab_word,
                        vocab_postag=vocab_postag,
                        vocab_deprel=vocab_deprel,
                        word_dim=word_dim,
                        postag_dim=postag_dim,
                        deprel_dim=deprel_dim,
                        lstm_dim=lstm_dim,
                        mlp_dim=mlp_dim,
                        initialW=initialW)
    else:
        raise ValueError("Invalid model_name=%s" % model_name)
    utils.writelog("Initialized the model ``%s''" % model_name)

    # Load pre-trained parameters
    if actiontype != "train":
        serializers.load_npz(path_snapshot, model)
        utils.writelog("Loaded trained parameters from %s" % path_snapshot)

    model.to_gpu(gpu)

    ####################
    # Decoder preparation
    decoder = decoders.IncrementalCKYDecoder()

    ####################
    # Initializer preparation
    sampler = treesamplers.TreeSampler(initial_tree_sampling.split("_"))

    ####################
    # Training / evaluation
    if actiontype == "train":
        with chainer.using_config("train", True):
            if dev_size > 0:
                # Training with cross validation
                train_dataset, dev_dataset = utils.split_dataset(dataset=train_dataset, n_dev=dev_size, seed=None)
                with open(os.path.join(config.getpath("results"), basename + ".valid_gold.ctrees"), "w") as f:
                    for data in dev_dataset:
                        f.write("%s\n" % " ".join(data.nary_sexp))
            else:
                # Training with the full training set
                dev_dataset = None

            if data_augmentation:
                train_dataset = np.concatenate([train_dataset, external_train_dataset], axis=0)

            train(
                model=model,
                decoder=decoder,
                sampler=sampler,
                max_epoch=max_epoch,
                n_init_epochs=n_init_epochs,
                negative_size=negative_size,
                batch_size=batch_size,
                weight_decay=weight_decay,
                gradient_clipping=gradient_clipping,
                optimizer_name=optimizer_name,
                train_dataset=train_dataset,
                dev_dataset=dev_dataset,
                path_train=path_train,
                path_valid=path_valid,
                path_snapshot=path_snapshot,
                path_pred=os.path.join(config.getpath("results"), basename + ".valid_pred.ctrees"),
                path_gold=os.path.join(config.getpath("results"), basename + ".valid_gold.ctrees"))

    elif actiontype == "evaluate":
        with chainer.using_config("train", False), chainer.no_backprop_mode():
            # Test
            parse(
                model=model,
                decoder=decoder,
                dataset=test_dataset,
                path_pred=path_pred)
            scores = metrics.rst_parseval(
                        pred_path=path_pred,
                        gold_path=os.path.join(config.getpath("data"), "rstdt", "wsj", "test", "gold.labeled.nary.ctrees"))
            old_scores = metrics.old_rst_parseval(
                        pred_path=path_pred,
                        gold_path=os.path.join(config.getpath("data"), "rstdt", "wsj", "test", "gold.labeled.nary.ctrees"))
            out = {
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
            utils.write_json(path_eval, out)
            utils.writelog(utils.pretty_format_dict(out))

    utils.writelog("Done: %s" % basename)

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
    :type model: SpanBasedModel
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
            parse(
                model=model,
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
                                        inputs=edu_ids,
                                        edus=edus,
                                        edus_head=edus_head,
                                        sbnds=sbnds,
                                        pbnds=pbnds)
                    else:
                        span_scores = precompute_all_span_scores(
                                                model=model,
                                                edus=edus,
                                                edus_postag=edus_postag,
                                                sbnds=sbnds,
                                                pbnds=pbnds,
                                                padded_edu_vectors=padded_edu_vectors,
                                                mask_bwd=mask_bwd,
                                                mask_fwd=mask_fwd)
                        pos_sexp = decoder.decode(
                                        span_scores=span_scores,
                                        inputs=edu_ids,
                                        sbnds=sbnds,
                                        pbnds=pbnds,
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
            loss_bracketing, acc_bracketing= 0.0, 0.0
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
                        span_scores = precompute_all_span_scores(
                                                model=model,
                                                edus=edus,
                                                edus_postag=edus_postag,
                                                sbnds=sbnds,
                                                pbnds=pbnds,
                                                padded_edu_vectors=padded_edu_vectors,
                                                mask_bwd=mask_bwd,
                                                mask_fwd=mask_fwd)
                        neg_bin_sexp = decoder.decode(
                                            span_scores=span_scores,
                                            inputs=edu_ids,
                                            sbnds=sbnds,
                                            pbnds=pbnds,
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
                    neg_bin_sexp = negative_tree_sampler.sample(inputs=edu_ids, sbnds=sbnds, pbnds=pbnds)
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

                # Bracketing Loss
                for neg_i in range(negative_size):
                    loss_bracketing += F.clip(pred_scores[1+neg_i] + margins[neg_i] - pred_scores[0], 0.0, 10000000.0)

                # Ranking Accuracy
                pred_scores = F.reshape(pred_scores, (1, 1+negative_size)) # (1, 1+negative_size)
                gold_scores = np.zeros((1,), dtype=np.int32) # (1,)
                gold_scores = utils.convert_ndarray_to_variable(gold_scores, seq=False) # (1,)
                acc_bracketing += F.accuracy(pred_scores, gold_scores)

                actual_batchsize += 1

            # Backward & Update
            actual_batchsize = float(actual_batchsize)
            loss_bracketing = loss_bracketing / actual_batchsize
            acc_bracketing = acc_bracketing / actual_batchsize
            loss = loss_bracketing
            model.zerograds()
            loss.backward()
            opt.update()
            it += 1

            # Write log
            loss_bracketing_data = float(cuda.to_cpu(loss_bracketing.data))
            acc_bracketing_data = float(cuda.to_cpu(acc_bracketing.data))
            out = {"iter": it,
                   "epoch": epoch,
                   "progress": "%d/%d" % (inst_i+actual_batchsize, n_train),
                   "progress_ratio": float(inst_i+actual_batchsize)/n_train*100.0,
                   "Bracketing Loss": loss_bracketing_data,
                   "Ranking Accuracy": acc_bracketing_data * 100.0}
            writer_train.write(out)
            utils.writelog(utils.pretty_format_dict(out))
        ########## M-Step (END) ##########

        if dev_dataset is not None:
            # Validation
            with chainer.using_config("train", False), chainer.no_backprop_mode():
                parse(
                    model=model,
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

def parse(model, decoder, dataset, path_pred):
    """
    :type model: SpanBasedModel
    :type decoder: IncrementalCKYDecoder
    :type dataset: numpy.ndarray
    :type path_pred: str
    :rtype: None
    """
    with open(path_pred, "w") as f:

        for data in pyprind.prog_bar(dataset):
            edu_ids = data.edu_ids
            edus = data.edus
            edus_postag = data.edus_postag
            edus_head = data.edus_head
            sbnds = data.sbnds
            pbnds = data.pbnds

            # Feature extraction
            edu_vectors = model.forward_edus(edus, edus_postag, edus_head) # (n_edus, bilstm_dim)
            padded_edu_vectors = model.pad_edu_vectors(edu_vectors) # (n_edus+2, bilstm_dim)
            mask_bwd, mask_fwd = model.make_masks() # (1, bilstm_dim), (1, bilstm_dim)

            # Parsing (bracketing)
            span_scores = precompute_all_span_scores(
                                    model=model,
                                    edus=edus,
                                    edus_postag=edus_postag,
                                    sbnds=sbnds,
                                    pbnds=pbnds,
                                    padded_edu_vectors=padded_edu_vectors,
                                    mask_bwd=mask_bwd,
                                    mask_fwd=mask_fwd)
            unlabeled_sexp = decoder.decode(
                                    span_scores=span_scores,
                                    inputs=edu_ids,
                                    sbnds=sbnds,
                                    pbnds=pbnds,
                                    use_sbnds=True,
                                    use_pbnds=True) # list of str
            unlabeled_tree = treetk.sexp2tree(unlabeled_sexp, with_nonterminal_labels=False, with_terminal_labels=False)
            unlabeled_tree.calc_spans()
            unlabeled_spans = treetk.aggregate_spans(unlabeled_tree, include_terminal=False, order="pre-order") # list of (int, int)

            # Parsing (assigning majority labels to the unlabeled tree)
            span2label = {(b,e): "<ELABORATION,N/S>" for (b,e) in unlabeled_spans}
            labeled_tree = treetk.assign_labels(unlabeled_tree, span2label, with_terminal_labels=False)
            labeled_sexp = treetk.tree2sexp(labeled_tree)

            f.write("%s\n" % " ".join(labeled_sexp))

def precompute_all_span_scores(
                        model,
                        edus,
                        edus_postag,
                        sbnds,
                        pbnds,
                        padded_edu_vectors,
                        mask_bwd,
                        mask_fwd):
    """
    :type model: SpanBasedModel
    :type edus: list of list of str
    :type edus_postag: list of list of str
    :type sbnds: list of (int, int)
    :type pbnds: list of (int, int)
    :type padded_edu_vectors: Variable(shape=(n_edus+2, bilstm_dim+tempfeat_dim), dtype=np.float32)
    :type mask_bwd: Variable(shape=(1, bilstm_dim), dtype=np.float32)
    :type mask_fwd: Variable(shape=(1, bilstm_dim), dtype=np.float32)
    :rtype: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
    """
    n_edus = len(edus)

    result = np.zeros(shape=(n_edus, n_edus), dtype="float")

    spans = []
    for d in range(1, n_edus):
        for i1 in range(0, n_edus - d):
            i3 = i1 + d
            spans.append((i1, i3))

    span_scores = model.forward_spans_for_bracketing(
                                edus=edus,
                                edus_postag=edus_postag,
                                sbnds=sbnds,
                                pbnds=pbnds,
                                padded_edu_vectors=padded_edu_vectors,
                                mask_bwd=mask_bwd,
                                mask_fwd=mask_fwd,
                                batch_spans=[spans],
                                aggregate=False) # (1, n_spans, bilstm_dim + tempfeat_dim)
    span_scores = cuda.to_cpu(span_scores.data)[0] # (n_spans, 1)

    for span_i, (i1, i3) in enumerate(spans):
        result[i1,i3] = span_scores[span_i,0]
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--initial_tree_sampling", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_augmentation", action="store_true")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--actiontype", type=str, required=True)
    parser.add_argument("--max_epoch", type=int, default=-1)
    parser.add_argument("--dev_size", type=int, default=0)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        utils.logger.error(e, exc_info=True)

