import argparse
import os
import random
import time

import chainer
from chainer import cuda, serializers
import numpy as np

import utils

import dataloader
import models
import treesamplers
import decoders
import training
import parsing
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

    train_databatch = dataloader.read_rstdt("train", relation_level="coarse-grained", with_root=False)
    test_databatch = dataloader.read_rstdt("test", relation_level="coarse-grained", with_root=False)
    vocab_word = utils.read_vocab(os.path.join(config.getpath("data"), "rstdt-vocab", "words.vocab.txt"))
    vocab_postag = utils.read_vocab(os.path.join(config.getpath("data"), "rstdt-vocab", "postags.vocab.txt"))
    vocab_deprel = utils.read_vocab(os.path.join(config.getpath("data"), "rstdt-vocab", "deprels.vocab.txt"))

    if data_augmentation:
        external_train_databatch = dataloader.read_ptbwsj_wo_rstdt(with_root=False)
        # Remove documents with only one leaf node
        filtering_function = lambda d,i: len(d.batch_edu_ids[i]) == 1
        external_train_databatch = utils.filter_databatch(external_train_databatch, filtering_function)

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
                                                databatch=train_databatch)
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
                train_databatch, dev_databatch = dataloader.randomsplit(
                                                        n_dev=dev_size,
                                                        databatch=train_databatch)
                with open(os.path.join(config.getpath("results"), basename + ".valid_gold.ctrees"), "w") as f:
                    for sexp in dev_databatch.batch_nary_sexp:
                        f.write("%s\n" % " ".join(sexp))
            else:
                # Training with the full training set
                dev_databatch = None

            if data_augmentation:
                train_databatch = utils.concat_databatch(
                                            train_databatch,
                                            external_train_databatch)
            training.train(
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
                train_databatch=train_databatch,
                dev_databatch=dev_databatch,
                path_train=path_train,
                path_valid=path_valid,
                path_snapshot=path_snapshot,
                path_pred=os.path.join(config.getpath("results"), basename + ".valid_pred.ctrees"),
                path_gold=os.path.join(config.getpath("results"), basename + ".valid_gold.ctrees"))

    elif actiontype == "evaluate":
        with chainer.using_config("train", False), chainer.no_backprop_mode():
            # Test
            parsing.parse(model=model,
                          decoder=decoder,
                          databatch=test_databatch,
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

