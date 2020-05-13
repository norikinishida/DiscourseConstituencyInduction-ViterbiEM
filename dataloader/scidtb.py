import os

import numpy as np

import utils
import treetk

def read_scidtb(split, sub_dir, relation_level):
    """
    :type split: str
    :type sub_dir: str
    :type relation_level: str
    :rtype: DataBatch
    """
    if not relation_level in ["coarse-grained", "fine-grained"]:
        raise ValueError("relation_level must be 'coarse-grained', or 'fine-grained'")
    if not sub_dir in ["", "gold", "second_annotate"]:
        raise ValueError("sub_dir must be '', 'gold', or 'second_annotate'")

    config = utils.Config()

    path_root = os.path.join(config.getpath("data"), "scidtb", "preprocessed", split, sub_dir)

    if relation_level == "coarse-grained":
        relation_mapper = treetk.rstdt.RelationMapper(corpus_name="scidtb")

    # Reading
    batch_edu_ids = []
    batch_edus = []
    batch_edus_postag = []
    batch_edus_head = []
    batch_sbnds = []
    batch_pbnds = []
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
        path_arcs = os.path.join(path_root, filename.replace(".edus.tokens", ".arcs"))

        # EDUs
        edus = utils.read_lines(path_edus, process=lambda line: line.split())
        edus = [["<root>"]] + edus
        batch_edus.append(edus)

        # EDU IDs
        edu_ids = np.arange(len(edus)).tolist()
        batch_edu_ids.append(edu_ids)

        # EDUs (POS tags)
        edus_postag = utils.read_lines(path_edus_postag, process=lambda line: line.split())
        edus_postag = [["<root>"]] + edus_postag
        batch_edus_postag.append(edus_postag)

        # EDUs (head)
        edus_head = utils.read_lines(path_edus_head, process=lambda line: tuple(line.split()))
        edus_head = [("<root>", "<root>", "<root>")] + edus_head
        batch_edus_head.append(edus_head)

        # Sentence boundaries
        sbnds = utils.read_lines(path_sbnds, process=lambda line: tuple([int(x) for x in line.split()]))
        batch_sbnds.append(sbnds)

        # Paragraph boundaries
        pbnds = utils.read_lines(path_pbnds, process=lambda line: tuple([int(x) for x in line.split()]))
        batch_pbnds.append(pbnds)

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
            == len(batch_arcs)
    # NOTE that sentence/paragraph boundaries do NOT consider ROOTs

    # Conversion to numpy.ndarray
    batch_edu_ids = np.asarray(batch_edu_ids, dtype="O")
    batch_edus = np.asarray(batch_edus, dtype="O")
    batch_edus_postag = np.asarray(batch_edus_postag, dtype="O")
    batch_edus_head = np.asarray(batch_edus_head, dtype="O")
    batch_sbnds = np.asarray(batch_sbnds, dtype="O")
    batch_pbnds = np.asarray(batch_pbnds, dtype="O")
    batch_arcs = np.asarray(batch_arcs, dtype="O")

    # Conversion to DataBatch
    databatch = utils.DataBatch(
                        batch_edu_ids=batch_edu_ids,
                        batch_edus=batch_edus,
                        batch_edus_postag=batch_edus_postag,
                        batch_edus_head=batch_edus_head,
                        batch_sbnds=batch_sbnds,
                        batch_pbnds=batch_pbnds,
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
        n_edus += len(edus[1:]) # Exclude the ROOT

    utils.writelog("split=%s, sub_dir=%s" % (split, sub_dir))
    utils.writelog("# of documents=%d" % n_docs)
    utils.writelog("# of paragraphs=%d" % n_paras)
    utils.writelog("# of sentences=%d" % n_sents)
    utils.writelog("# of EDUs (w/o ROOTs)=%d" % n_edus)
    return databatch

