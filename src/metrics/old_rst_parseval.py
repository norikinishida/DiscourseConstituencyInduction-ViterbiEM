import multiset

import utils
import treetk

def old_rst_parseval(pred_path, gold_path):
    """
    :type pred_path: str
    :type gold_path: str
    :rtype: {str: {str: Any}}
    """
    preds = read_trees(pred_path)
    golds = read_trees(gold_path)

    input_lengths = []
    for pred_tree, gold_tree in zip(preds, golds):
        pred_len = len(pred_tree.leaves())
        gold_len = len(gold_tree.leaves())
        assert pred_len == gold_len
        input_lengths.append(pred_len)

    preds = [get_spans(tree) for tree in preds]
    golds = [get_spans(tree) for tree in golds]
    scores = _old_rst_parseval(preds, golds, input_lengths)
    return scores

def read_trees(path):
    """
    :type path: str
    :rtype: list of NonTerminal
    """
    sexps = utils.read_lines(path, process=lambda line: line.split())
    trees = [treetk.rstdt.postprocess(treetk.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=False)) for sexp in sexps]
    return trees

def get_spans(tree):
    """
    :type tree: NonTerminal
    :rtype: list of (int, int, str, str)
    """
    tree.calc_spans()
    spans = treetk.aggregate_spans(tree, include_terminal=False, order="pre-order")
    new_spans = []
    for begin_i, end_i, label in spans:
        relation_label, nuclearity_label = treetk.rstdt.extract_relation_and_nuclearity_labels(label)
        new_span = (begin_i, end_i, relation_label, nuclearity_label)
        new_spans.append(new_span)
    return new_spans

def _old_rst_parseval(preds, golds, input_lengths):
    """
    :type preds: list of list of (int, int, str, str)
    :type golds: list of list of (int, int, str, str)
    :type input_lengths: list of int
    :rtype: {str: {str: Any}}
    """
    assert len(preds) == len(golds) == len(input_lengths)

    scores = {} # {str: {str: Any}}

    total_ok_dict = {}
    total_pred_dict = {}
    total_gold_dict = {}
    for key in ["S", "S+N", "S+R", "S+N+R"]:
        total_ok_dict[key] = 0.0
        total_pred_dict[key] = 0.0
        total_gold_dict[key] = 0.0

    for pred_spans, gold_spans, input_length in zip(preds, golds, input_lengths):
        pred_spans_dict = {}
        gold_spans_dict = {}
        pred_spans_dict["S"] = [(b,e) for b,e,r,n in pred_spans]
        gold_spans_dict["S"] = [(b,e) for b,e,r,n in gold_spans]
        pred_spans_dict["S+N"] = [(b,e,n) for b,e,r,n in pred_spans]
        gold_spans_dict["S+N"] = [(b,e,n) for b,e,r,n in gold_spans]
        pred_spans_dict["S+R"] = [(b,e,r) for b,e,r,n in pred_spans]
        gold_spans_dict["S+R"] = [(b,e,r) for b,e,r,n in gold_spans]
        pred_spans_dict["S+N+R"] = pred_spans
        gold_spans_dict["S+N+R"] = gold_spans

        for key in ["S", "S+N", "S+R", "S+N+R"]:
            a = multiset.Multiset(pred_spans_dict[key])
            b = multiset.Multiset(gold_spans_dict[key])
            n_ok = float(len(a & b))
            n_pred = float(len(a))
            n_gold = float(len(b))

            # NOTE: Pre-terminal spans (i.e., spans of length=1) cannot be incorrect in unlabeled constituency metrics
            if key == "S":
                n_ok += input_length
                n_pred += input_length
                n_gold += input_length

            total_ok_dict[key] += n_ok
            total_pred_dict[key] += n_pred
            total_gold_dict[key] += n_gold

    for key in ["S", "S+N", "S+R", "S+N+R"]:
        precision = float(total_ok_dict[key]) / float(total_pred_dict[key])
        recall = float(total_ok_dict[key]) / float(total_gold_dict[key])
        f1 = (2 * precision * recall) / (precision + recall)
        precision_info = "%d/%d" % (total_ok_dict[key], total_pred_dict[key])
        recall_info = "%d/%d" % (total_ok_dict[key], total_gold_dict[key])
        scores[key] = {"Precision": precision,
                       "Recall": recall,
                       "Micro F1": f1,
                       "Precision_info": precision_info,
                       "Recall_info": recall_info}

    return scores

