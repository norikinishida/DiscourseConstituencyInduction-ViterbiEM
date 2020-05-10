import os

import numpy as np

import utils
import treetk

def read_rstdt(split, relation_level, with_root=False):
    """
    :type split: str
    :type relation_level: str
    :type with_root: bool
    :rtype: DataBatch
    """
    if not relation_level in ["coarse-grained", "fine-grained"]:
        raise ValueError("relation_level must be 'coarse-grained' or 'fine-grained'")

    config = utils.Config()

    path_root = os.path.join(config.getpath("data"), "rstdt", "wsj", split)

    if relation_level == "coarse-grained":
        relation_mapper = treetk.rstdt.RelationMapper()

    # Reading
    batch_edu_ids = []
    batch_edus = []
    batch_edus_postag = []
    batch_edus_head = []
    batch_sbnds = []
    batch_pbnds = []
    batch_nary_sexp = []
    batch_bin_sexp = []
    batch_arcs = []

    filenames = os.listdir(path_root)
    filenames = [n for n in filenames if n.endswith(".edus.tokens")]
    filenames.sort()

    for filename in filenames:
        # Path
        path_edus = os.path.join(path_root, filename + ".preprocessed")
        path_edus_postag = os.path.join(path_root, filename.replace(".edus.tokens", ".edus.postags"))
        path_edus_head = os.path.join(path_root, filename.replace(".edus.tokens", ".edus.heads"))
        path_sbnds = os.path.join(path_root, filename.replace(".edus.tokens", ".sbnds"))
        path_pbnds = os.path.join(path_root, filename.replace(".edus.tokens", ".pbnds"))
        path_nary_sexp = os.path.join(path_root, filename.replace(".edus.tokens", ".labeled.nary.ctree"))
        path_bin_sexp = os.path.join(path_root, filename.replace(".edus.tokens", ".labeled.bin.ctree"))
        path_arcs = os.path.join(path_root, filename.replace(".edus.tokens", ".arcs"))

        # EDUs
        edus = utils.read_lines(path_edus, process=lambda line: line.split())
        if with_root:
            edus = [["<root>"]] + edus
        batch_edus.append(edus)

        # EDU IDs
        edu_ids = np.arange(len(edus)).tolist()
        batch_edu_ids.append(edu_ids)

        # EDUs (POS tags)
        edus_postag = utils.read_lines(path_edus_postag, process=lambda line: line.split())
        if with_root:
            edus_postag = [["<root>"]] + edus_postag
        batch_edus_postag.append(edus_postag)

        # EDUs (head)
        edus_head = utils.read_lines(path_edus_head, process=lambda line: tuple(line.split()))
        if with_root:
            edus_head = [("<root>", "<root>", "<root>")] + edus_head
        batch_edus_head.append(edus_head)

        # Sentence boundaries
        sbnds = utils.read_lines(path_sbnds, process=lambda line: tuple([int(x) for x in line.split()]))
        batch_sbnds.append(sbnds)

        # Paragraph boundaries
        pbnds = utils.read_lines(path_pbnds, process=lambda line: tuple([int(x) for x in line.split()]))
        batch_pbnds.append(pbnds)

        # Constituent tree
        nary_sexp = utils.read_lines(path_nary_sexp, process=lambda line: line.split())[0]
        bin_sexp = utils.read_lines(path_bin_sexp, process=lambda line: line.split())[0]
        if relation_level == "coarse-grained":
            nary_tree = treetk.rstdt.postprocess(treetk.sexp2tree(nary_sexp, with_nonterminal_labels=True, with_terminal_labels=False))
            bin_tree = treetk.rstdt.postprocess(treetk.sexp2tree(bin_sexp, with_nonterminal_labels=True, with_terminal_labels=False))
            nary_tree = treetk.rstdt.map_relations(nary_tree, mode="f2c")
            bin_tree = treetk.rstdt.map_relations(bin_tree, mode="f2c")
            nary_sexp = treetk.tree2sexp(nary_tree)
            bin_sexp = treetk.tree2sexp(bin_tree)
        batch_nary_sexp.append(nary_sexp)
        batch_bin_sexp.append(bin_sexp)

        # Dependency tree
        hyphens = utils.read_lines(path_arcs, process=lambda line: line.split())
        assert len(hyphens) == 1
        hyphens = hyphens[0] # list of str
        arcs = treetk.hyphens2arcs(hyphens) # list of (int, int, str)
        if relation_level == "coarse-grained":
            arcs = [(h,d,relation_mapper.f2c(l)) for h,d,l in arcs]
        batch_arcs.append(arcs)

    assert len(batch_edu_ids) \
            == len(batch_edus) \
            == len(batch_edus_postag) \
            == len(batch_edus_head) \
            == len(batch_sbnds) \
            == len(batch_pbnds) \
            == len(batch_nary_sexp) \
            == len(batch_bin_sexp) \
            == len(batch_arcs)
    # NOTE that sentence/paragraph boundaries do NOT consider ROOTs even if with_root=True.

    # Conversion to numpy.ndarray
    batch_edu_ids = np.asarray(batch_edu_ids, dtype="O")
    batch_edus = np.asarray(batch_edus, dtype="O")
    batch_edus_postag = np.asarray(batch_edus_postag, dtype="O")
    batch_edus_head = np.asarray(batch_edus_head, dtype="O")
    batch_sbnds = np.asarray(batch_sbnds, dtype="O")
    batch_pbnds = np.asarray(batch_pbnds, dtype="O")
    batch_nary_sexp = np.asarray(batch_nary_sexp, dtype="O")
    batch_bin_sexp = np.asarray(batch_bin_sexp, dtype="O")
    batch_arcs = np.asarray(batch_arcs, dtype="O")

    # Conversion to DataBatch
    databatch = utils.DataBatch(
                        batch_edu_ids=batch_edu_ids,
                        batch_edus=batch_edus,
                        batch_edus_postag=batch_edus_postag,
                        batch_edus_head=batch_edus_head,
                        batch_sbnds=batch_sbnds,
                        batch_pbnds=batch_pbnds,
                        batch_nary_sexp=batch_nary_sexp,
                        batch_bin_sexp=batch_bin_sexp,
                        batch_arcs=batch_arcs)

    n_docs = len(databatch)

    n_paras = 0
    for pbnds in batch_pbnds:
        n_paras += len(pbnds)

    n_sents = 0
    for sbnds in batch_sbnds:
        n_sents += len(sbnds)

    n_edus = 0
    for edus in batch_edus:
        if with_root:
            n_edus += len(edus[1:]) # Exclude the ROOT
        else:
            n_edus += len(edus)

    utils.writelog("split=%s" % split)
    utils.writelog("# of documents=%d" % n_docs)
    utils.writelog("# of paragraphs=%d" % n_paras)
    utils.writelog("# of sentences=%d" % n_sents)
    utils.writelog("# of EDUs (w/o ROOTs)=%d" % n_edus)
    return databatch

