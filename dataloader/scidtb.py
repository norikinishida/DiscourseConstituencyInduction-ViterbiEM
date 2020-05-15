import os

import numpy as np

import utils
import treetk

def read_scidtb(split, sub_dir, relation_level):
    """
    :type split: str
    :type sub_dir: str
    :type relation_level: str
    :rtype: numpy.ndarray(shape=(dataset_size,), dtype="O")
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
    dataset = []

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

        kargs = {}

        # EDUs
        edus = utils.read_lines(path_edus, process=lambda line: line.split())
        edus = [["<root>"]] + edus
        kargs["edus"] = edus

        # EDU IDs
        edu_ids = np.arange(len(edus)).tolist()
        kargs["edu_ids"] = edu_ids

        # EDUs (POS tags)
        edus_postag = utils.read_lines(path_edus_postag, process=lambda line: line.split())
        edus_postag = [["<root>"]] + edus_postag
        kargs["edus_postag"] = edus_postag

        # EDUs (head)
        edus_head = utils.read_lines(path_edus_head, process=lambda line: tuple(line.split()))
        edus_head = [("<root>", "<root>", "<root>")] + edus_head
        kargs["edus_head"] = edus_head

        # Sentence boundaries
        sbnds = utils.read_lines(path_sbnds, process=lambda line: tuple([int(x) for x in line.split()]))
        kargs["sbnds"] = sbnds

        # Paragraph boundaries
        pbnds = utils.read_lines(path_pbnds, process=lambda line: tuple([int(x) for x in line.split()]))
        kargs["pbnds"] = pbnds

        # Dependency tree
        hyphens = utils.read_lines(path_arcs, process=lambda line: line.split())
        assert len(hyphens) == 1
        hyphens = hyphens[0] # list of str
        arcs = treetk.hyphens2arcs(hyphens) # list of (int, int, str)
        if relation_level == "coarse-grained":
            arcs = [(h,d,relation_mapper.f2c(l)) for h,d,l in arcs]
        kargs["arcs"] = arcs

        data = utils.DataInstance(**kargs)
        dataset.append(data)

   # NOTE that sentence/paragraph boundaries do NOT consider ROOTs

    dataset = np.asarray(dataset, dtype="O")

    n_docs = len(dataset)

    n_paras = 0
    for data in dataset:
        n_paras += len(data.pbnds)

    n_sents = 0
    for data in dataset:
        n_sents += len(data.sbnds)

    n_edus = 0
    for data in dataset:
        n_edus += len(data.edus[1:]) # Exclude the ROOT

    utils.writelog("split=%s, sub_dir=%s" % (split, sub_dir))
    utils.writelog("# of documents=%d" % n_docs)
    utils.writelog("# of paragraphs=%d" % n_paras)
    utils.writelog("# of sentences=%d" % n_sents)
    utils.writelog("# of EDUs (w/o ROOTs)=%d" % n_edus)
    return dataset

