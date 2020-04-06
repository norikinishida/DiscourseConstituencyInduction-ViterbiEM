import os

import utils
import treetk
import treetk.rstdt

def main():
    config = utils.Config()

    # NOTE: We use n-ary ctrees (ie., *.labeled.nary.ctree) to generate dtrees.
    #       Morey et al. (2018) demonstrate that scores evaluated on these dtrees are
    #       superficially lower than those on right-heavy binarized trees (ie., *.labeled.bin.ctree).

    filenames = os.listdir(os.path.join(config.getpath("data"), "rstdt", "renamed"))
    filenames = [n for n in filenames if n.endswith(".labeled.nary.ctree")]
    filenames.sort()

    def func_label_rule(node, i, j):
        if len(node.relations) == 1:
            return node.relations[0] # Left-most node is head.
        else:
            if i > j:
                return node.relations[j]
            else:
                return node.relations[j-1]

    for filename in filenames:
        sexp = utils.read_lines(
                    os.path.join(config.getpath("data"), "rstdt", "renamed", filename),
                    process=lambda line: line.split())
        assert len(sexp) == 1
        sexp = sexp[0]

        # Constituency
        ctree = treetk.rstdt.postprocess(treetk.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=False))

        # Dependency
        # Assign heads
        ctree = treetk.rstdt.assign_heads(ctree)
        # Conversion
        dtree = treetk.ctree2dtree(ctree, func_label_rule=func_label_rule)
        arcs = dtree.tolist(labeled=True)

        # Write
        with open(os.path.join(
                    config.getpath("data"), "rstdt", "renamed",
                    filename.replace(".labeled.nary.ctree", ".arcs")), "w") as f:
            f.write("%s\n" % " ".join(["%d-%d-%s" % (h,d,l) for h,d,l in arcs]))

if __name__ == "__main__":
    main()
