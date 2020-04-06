import os

import numpy as np

import utils

def read_ptbwsj_wo_rstdt(with_root):
    """
    :type with_root: bool
    :rtype: DataBatch
    """
    config = utils.Config()

    path_root = os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed")

    # Reading
    batch_edu_ids = []
    batch_edus = []
    batch_edus_postag = []
    batch_edus_head = []
    batch_sbnds = []
    batch_pbnds = []
    filenames = os.listdir(os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed"))
    filenames = [n for n in filenames if n.endswith(".paragraph.boundaries")]
    filenames = [n.replace(".paragraph.boundaries", ".edus") for n in filenames]
    filenames.sort()
    for filename in filenames:
        # Path
        path_edus = os.path.join(path_root, filename + ".preprocessed")
        path_edus_postag = os.path.join(path_root, filename + ".postags")
        path_edus_head = os.path.join(path_root, filename + ".heads")
        path_sbnds = os.path.join(path_root, filename.replace(".edus", ".sentence.proj.boundaries"))
        path_pbnds = os.path.join(path_root, filename.replace(".edus", ".paragraph.boundaries"))
        # EDUs
        edus = utils.read_lines(path_edus, process=lambda line: line.split())
        if with_root:
            edus = [["<root>"]] + edus
        batch_edus.append(edus)
        # EDU IDs
        edu_ids = np.arange(len(edus)).tolist()
        batch_edu_ids.append(edu_ids)
        # EDUs (Syntactic features; POSTAG)
        edus_postag = utils.read_lines(path_edus_postag, process=lambda line: line.split())
        if with_root:
            edus_postag = [["<root>"]] + edus_postag
        batch_edus_postag.append(edus_postag)
        # EDUs (Syntactic features; HEAD)
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
    assert len(batch_edu_ids) \
            == len(batch_edus) \
            == len(batch_edus_postag) \
            == len(batch_edus_head) \
            == len(batch_sbnds) \
            == len(batch_pbnds)

    # Conversion to numpy.ndarray
    batch_edu_ids = np.asarray(batch_edu_ids, dtype="O")
    batch_edus = np.asarray(batch_edus, dtype="O")
    batch_edus_postag = np.asarray(batch_edus_postag, dtype="O")
    batch_edus_head = np.asarray(batch_edus_head, dtype="O")
    batch_sbnds = np.asarray(batch_sbnds, dtype="O")
    batch_pbnds = np.asarray(batch_pbnds, dtype="O")

    # Conversion to DataBatch
    databatch = utils.DataBatch(
                        batch_edu_ids=batch_edu_ids,
                        batch_edus=batch_edus,
                        batch_edus_postag=batch_edus_postag,
                        batch_edus_head=batch_edus_head,
                        batch_sbnds=batch_sbnds,
                        batch_pbnds=batch_pbnds)

    total_edus = 0
    for edus in batch_edus:
        if with_root:
            total_edus += len(edus[1:]) # Exclude the ROOT
        else:
            total_edus += len(edus)
    utils.writelog("# of instances=%d" % len(databatch))
    utils.writelog("# of EDUs (w/o ROOTs)=%d" % total_edus)
    return databatch

