import os

import numpy as np

import utils

def read_ptbwsj_wo_rstdt(with_root=False):
    """
    :type with_root: bool
    :rtype: numpy.ndarray(shape=(dataset_size,), dtype="O")
    """
    config = utils.Config()

    path_root = os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt")

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

        kargs = {}

        # EDUs
        edus = utils.read_lines(path_edus, process=lambda line: line.split())
        if with_root:
            edus = [["<root>"]] + edus
        kargs["edus"] = edus

        # EDU IDs
        edu_ids = np.arange(len(edus)).tolist()
        kargs["edu_ids"] = edu_ids

        # EDUs (POS tags)
        edus_postag = utils.read_lines(path_edus_postag, process=lambda line: line.split())
        if with_root:
            edus_postag = [["<root>"]] + edus_postag
        kargs["edus_postag"] = edus_postag

        # EDUs (head)
        edus_head = utils.read_lines(path_edus_head, process=lambda line: tuple(line.split()))
        if with_root:
            edus_head = [("<root>", "<root>", "<root>")] + edus_head
        kargs["edus_head"] = edus_head

        # Sentence boundaries
        sbnds = utils.read_lines(path_sbnds, process=lambda line: tuple([int(x) for x in line.split()]))
        kargs["sbnds"] = sbnds

        # Paragraph boundaries
        pbnds = utils.read_lines(path_pbnds, process=lambda line: tuple([int(x) for x in line.split()]))
        kargs["pbnds"] = pbnds

        data = utils.DataInstance(**kargs)
        dataset.append(data)

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
        if with_root:
            n_edus += len(data.edus[1:]) # Exclude the ROOT
        else:
            n_edus += len(data.edus)

    utils.writelog("# of documents=%d" % n_docs)
    utils.writelog("# of paragraphs=%d" % n_paras)
    utils.writelog("# of sentences=%d" % n_sents)
    utils.writelog("# of EDUs (w/o ROOTs)=%d" % n_edus)
    return dataset

