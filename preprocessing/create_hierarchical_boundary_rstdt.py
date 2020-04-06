import os

import numpy as np

import utils

from create_sentence_boundary_rstdt import test_boundaries
from create_sentence_boundary_rstdt import write_boundaries

def project_pbnds_to_sbnds(sbnds, pbnds):
    """
    :type sbnds: list of (int, int)
    :type pbnds: list of (int, int)
    :rtype: list of (int, int), int
    """
    max_id = -1
    for p_begin_i, p_end_i in pbnds:
        if p_end_i > max_id:
            max_id = p_end_i
    n_edus = max_id + 1

    memo = [{"BEGIN": False, "END": False} for i in range(0, n_edus)]
    memo = np.asarray(memo, dtype="O")

    # bnds の登録
    for p_begin_i, p_end_i in pbnds:
        memo[p_begin_i]["BEGIN"] = True
        memo[p_end_i]["END"] = True
    for s_begin_i, s_end_i in sbnds:
        memo[s_begin_i]["BEGIN"] = True
        memo[s_end_i]["END"] = True

    # New sentence boundary の計算
    result = []
    begin_i = 0
    while True:
        assert memo[begin_i]["BEGIN"]
        end_i = begin_i
        while True:
            if memo[end_i]["END"]:
                break
            else:
                end_i += 1
        result.append((begin_i, end_i))
        begin_i = end_i + 1
        if begin_i >= n_edus:
            break
    return result, n_edus

def replace_subtrees_with_ids(sbnds, pbnds):
    """
    :type sbnds: list of tuple of int
    :type pbnds: list of tuple of int
    :rtype: list of tuple of int
    """
    result = []
    p_i = 0
    span = []
    for s_i in range(len(sbnds)):
        s_begin_i, s_end_i = sbnds[s_i]
        p_begin_i, p_end_i = pbnds[p_i]
        if p_begin_i <= s_begin_i <= s_end_i <= p_end_i:
            span.append(s_i)
        else:
            result.append((min(span), max(span)))
            p_i += 1
            span = [s_i]
    if len(span) != 0:
        result.append((min(span), max(span)))
    assert len(result) == len(pbnds)
    return result

def main():
    config = utils.Config()

    filenames = os.listdir(os.path.join(config.getpath("data"), "rstdt", "renamed"))
    filenames = [n for n in filenames if n.endswith(".edus")]
    filenames.sort()

    for file_i, filename in enumerate(filenames):
        path_s = os.path.join(config.getpath("data"), "rstdt", "tmp.preprocessing",
                              filename.replace(".edus", ".sentence.boundaries"))
        path_p = os.path.join(config.getpath("data"), "rstdt", "tmp.preprocessing",
                              filename.replace(".edus", ".paragraph.boundaries"))

        sbnds = utils.read_lines(path_s, process=lambda l: tuple([int(x) for x in l.split()]))
        pbnds = utils.read_lines(path_p, process=lambda l: tuple([int(x) for x in l.split()]))

        sbnds_proj, n_edus = project_pbnds_to_sbnds(sbnds=sbnds, pbnds=pbnds)
        if sbnds != sbnds_proj:
            print("Projected paragraph boundaries into the sentence boundaries (+%d): %s" % \
                    (len(sbnds_proj) - len(sbnds), path_s))

        test_boundaries(sbnds_proj, n_edus)

        pbnds = replace_subtrees_with_ids(sbnds=sbnds_proj, pbnds=pbnds)


        write_boundaries(sbnds,
                         os.path.join(
                            config.getpath("data"), "rstdt", "renamed",
                            filename.replace(".edus", ".sentence.noproj.boundaries")))
        write_boundaries(sbnds_proj,
                         os.path.join(
                            config.getpath("data"), "rstdt", "renamed",
                            filename.replace(".edus", ".sentence.proj.boundaries")))
        write_boundaries(pbnds,
                         os.path.join(
                             config.getpath("data"), "rstdt", "renamed",
                             filename.replace(".edus", ".paragraph.boundaries")))

if __name__ == "__main__":
    main()
