import argparse
import os
import time

import numpy as np
import chainer
from chainer import cuda
import pyprind

import utils
import treetk

import dataloader
import treesamplers # NOTE
import metrics

def main(args):

    ####################
    # Arguments
    tree_sampling = args.tree_sampling # NOTE
    trial_name = args.name

    # Check
    assert len(tree_sampling.split("_")) == 3
    for type_ in tree_sampling.split("_"):
        assert type_ in ["X", "BU", "TD", "RB", "LB", "RB2"]
    assert tree_sampling.split("_")[2] != "X"
    assert tree_sampling.split("_")[1] != "RB2"
    assert tree_sampling.split("_")[2] != "RB2"

    if trial_name is None or trial_name == "None":
        trial_name = utils.get_current_time()

    ####################
    # Path setting
    config = utils.Config()

    basename = "%s.%s" \
            % (tree_sampling,
               trial_name)

    utils.mkdir(os.path.join(config.getpath("results"), "baselines"))
    path_log = os.path.join(config.getpath("results"), "baselines", basename + ".evaluation.log")
    path_pred = os.path.join(config.getpath("results"), "baselines", basename + ".evaluation.ctrees")
    path_eval = os.path.join(config.getpath("results"), "baselines", basename + ".evaluation.json")

    utils.set_logger(path_log)

    ####################
    # Random seed
    random_seed = trial_name
    random_seed = utils.hash_string(random_seed)
    np.random.seed(random_seed)
    cuda.cupy.random.seed(random_seed)

    ####################
    # Log so far
    utils.writelog("tree_sampling=%s" % tree_sampling)
    utils.writelog("trial_name=%s" % trial_name)

    utils.writelog("path_log=%s" % path_log)
    utils.writelog("path_pred=%s" % path_pred)
    utils.writelog("path_eval=%s" % path_eval)

    utils.writelog("random_seed=%d" % random_seed)

    ####################
    # Data preparation
    begin_time = time.time()

    test_databatch = dataloader.read_rstdt("test", relation_level="coarse-grained", with_root=False)

    end_time = time.time()
    utils.writelog("Loaded the corpus. %f [sec.]" % (end_time - begin_time))

    ####################
    # Tree-sampler preparation
    sampler = treesamplers.TreeSampler(tree_sampling.split("_")) # NOTE

    with chainer.using_config("train", False), chainer.no_backprop_mode():
        parse(sampler=sampler,
              databatch=test_databatch,
              path_pred=path_pred)
        scores = metrics.rst_parseval(
                    pred_path=path_pred,
                    gold_path=os.path.join(config.getpath("data"), "rstdt", "renamed", "test.labeled.nary.ctrees"))
        old_scores = metrics.old_rst_parseval(
                    pred_path=path_pred,
                    gold_path=os.path.join(config.getpath("data"), "rstdt", "renamed", "test.labeled.nary.ctrees"))
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

    utils.writelog("Done.")

def parse(sampler, databatch, path_pred):
    """
    :type sampler: TreeSampler
    :type databatch: DataBatch
    :type path_pred: str
    :rtype: None
    """
    with open(path_pred, "w") as f:
        prog_bar = pyprind.ProgBar(len(databatch))

        for edu_ids, edus, edus_postag, edus_head, sbnds, pbnds \
                in zip(databatch.batch_edu_ids,
                       databatch.batch_edus,
                       databatch.batch_edus_postag,
                       databatch.batch_edus_head,
                       databatch.batch_sbnds,
                       databatch.batch_pbnds):

            # Tree sampling (constituency)
            unlabeled_sexp = sampler.sample(
                                    sexps=edu_ids,
                                    edus=edus,
                                    edus_head=edus_head,
                                    sbnds=sbnds,
                                    pbnds=pbnds)
            unlabeled_tree = treetk.sexp2tree(unlabeled_sexp, with_nonterminal_labels=False, with_terminal_labels=False)
            unlabeled_tree.calc_spans()
            unlabeled_spans = treetk.aggregate_spans(unlabeled_tree, include_terminal=False, order="pre-order") # list of (int, int)

            # Assigning majority labels to the unlabeled tree
            span2label = {(b,e): "<ELABORATION,N/S>" for (b,e) in unlabeled_spans}
            labeled_tree = treetk.assign_labels(unlabeled_tree, span2label, with_terminal_labels=False)
            labeled_sexp = treetk.tree2sexp(labeled_tree)

            f.write("%s\n" % " ".join(labeled_sexp))

            prog_bar.update()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree_sampling", type=str, required=True)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        utils.logger.error(e, exc_info=True)

