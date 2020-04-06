import os

import utils
import treetk

def extract_head(arcs):
    """
    :type arcs: list of (int, int, str)
    :rtype: int, str
    """
    deps = [d for h,d,l in arcs]
    min_span = min(deps)
    max_span = max(deps)

    # Root's dependent?
    for arc in arcs:
        head, dep, label = arc
        if head == 0:
            assert label == "ROOT" # FIXME
            return dep, label

    # The left-most token that has a head out of the region
    for arc in arcs:
        head, dep, label = arc
        if (head < min_span) or (max_span < head):
            return dep, label

    raise ValueError("Should never happen: arcs=%s" % arcs)

def main():
    config = utils.Config()

    filenames = os.listdir(os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed"))
    filenames = [n for n in filenames if n.endswith(".paragraph.boundaries")]
    filenames = [n.replace(".paragraph.boundaries", ".edus") for n in filenames]
    filenames.sort()

    for filename in filenames:
        # Path
        path_edus = os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed", filename + ".preprocessed")
        path_postags = os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed", filename + ".postags")
        path_conll = os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "tmp.preprocessing", filename.replace(".edus", ".sentences.conll"))
        path_sbnds = os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed", filename.replace(".edus", ".sentence.noproj.boundaries"))
        path_out1 = os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed", filename + ".arcs") # e.g., "train.123.edus.arcs" (!= "train.123.arcs")
        path_out2 = os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed", filename + ".heads")

        # Read & Write
        with open(path_out1, "w") as f1, open(path_out2, "w") as f2:
            edus = utils.read_lines(path_edus, process=lambda line: line.split()) # list of list of str
            edus_postag = utils.read_lines(path_postags, process=lambda line: line.split()) # list of list of str
            sentences = utils.read_conll(path_conll, keys=["ID", "FORM", "LEMMA", "POSTAG", "_1", "HEAD", "DEPREL"]) # list of list of {str: str}
            sbnds = utils.read_lines(path_sbnds, process=lambda line: (int(x) for x in line.split())) # list of (int, int)
            assert len(sentences) == len(sbnds)

            for sentence, sbnd in zip(sentences, sbnds):
                begin_i, end_i = sbnd

                # EDUs for this sentence
                part_edus = edus[begin_i:end_i+1] # list of list of str
                part_edus_postag = edus_postag[begin_i:end_i+1] # list of list of str

                # Dependency tree for this sentence
                arcs = [] # list of (int, int, str)
                for conll_line in sentence:
                    assert len(conll_line) == 7
                    head = int(conll_line["HEAD"])
                    dep = int(conll_line["ID"])
                    label = conll_line["DEPREL"]
                    arcs.append((head, dep, label))
                dtree = treetk.arcs2dtree(arcs=arcs)

                # Write dependency trees (and heads) for each EDU
                token_i = 1 # Excluding ROOT
                for edu, edu_postag in zip(part_edus, part_edus_postag):
                    assert len(edu) == len(edu_postag)

                    # Dependency tree for this EDU
                    part_arcs = [] # list of (int, int, str)
                    for dep in range(token_i, token_i + len(edu)):
                        head, label = dtree.get_head(dep)
                        part_arcs.append((head, dep, label))
                    f1.write("%s\n" % " ".join(["%d-%d-%s" % (h,d,l) for h,d,l in part_arcs]))

                    # Head word/POS-tag of this EDU
                    edu_head, label = extract_head(part_arcs)
                    f2.write("%s %s %s\n" % (edu[edu_head - token_i], edu_postag[edu_head - token_i], label))

                    token_i += len(edu)

                assert token_i == len(dtree.tokens)

if __name__ == "__main__":
    main()
