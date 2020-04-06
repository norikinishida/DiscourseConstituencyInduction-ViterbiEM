import os

import utils
import treetk

def process1(path_in, path_out):
    """
    :type path_in: str
    :type path_out: str
    :rtype: None
    """
    utils.mkdir(path_out)

    filenames = os.listdir(path_in)
    filenames = [n for n in filenames if n.endswith(".edus")]
    filenames.sort()

    for file_i, filename in enumerate(filenames):
        # Read and write one document
        with open(os.path.join(path_out, filename), "w") as f:
            for line in open(os.path.join(path_in, filename)):
                line = line.strip()
                f.write("%s\n" % line)

        # Read and write the trees
        with open(os.path.join(
                    path_out,
                    filename.replace(".edus", ".labeled.nary.ctree")), "w") as f_labeled_nary, \
             open(os.path.join(
                    path_out,
                    filename.replace(".edus", ".unlabeled.nary.ctree")), "w") as f_unlabeled_nary, \
             open(os.path.join(
                    path_out,
                    filename.replace(".edus", ".labeled.bin.ctree")), "w") as f_labeled_bin, \
             open(os.path.join(
                    path_out,
                    filename.replace(".edus", ".unlabeled.bin.ctree")), "w") as f_unlabeled_bin:
            sexp = treetk.rstdt.read_sexp(os.path.join(path_in, filename.replace(".edus", ".dis")))
            tree = treetk.rstdt.sexp2tree(sexp)
            tree = treetk.rstdt.shift_labels(tree)
            f_labeled_nary.write("%s\n" % treetk.rstdt.tree2str(tree, labeled=True))
            f_unlabeled_nary.write("%s\n" % treetk.rstdt.tree2str(tree, labeled=False))

            # Binarize
            tree = treetk.rstdt.binarize(tree)
            tree = treetk.rstdt.shift_labels(tree)
            f_labeled_bin.write("%s\n" % treetk.rstdt.tree2str(tree, labeled=True))
            f_unlabeled_bin.write("%s\n" % treetk.rstdt.tree2str(tree, labeled=False))

def process2(path_in, path_out, split):
    """
    :type path_in: str
    :type path_out: str
    :type split: str
    :rtype: None
    """
    utils.mkdir(path_out)

    filenames = os.listdir(path_in)
    filenames = [n for n in filenames if n.endswith(".edus")]
    filenames.sort() # NOTE: Important

    with open(os.path.join(path_out, "filename_map.txt"), "a") as f_map:
        for file_i, filename in enumerate(filenames):
            new_filename = "%s.%03d" % (split, file_i)

            # Add to the filename map
            f_map.write("%s %s\n" %\
                    (os.path.join(path_in, filename),
                     os.path.join(path_out, new_filename + ".edus")))

            # Read and write one document (i.e., sequence of EDUs)
            edus = [] # list of str
            for line in open(os.path.join(path_in, filename)):
                edu = line.strip()
                edus.append(edu)
            with open(os.path.join(path_out, new_filename + ".edus"), "w") as f:
                for edu in edus:
                    f.write("%s\n" % edu)

            # Read and write the trees
            for ext in [".labeled.nary.ctree", ".unlabeled.nary.ctree",
                        ".labeled.bin.ctree", ".unlabeled.bin.ctree"]:
                sexp = utils.read_lines(os.path.join(path_in, filename.replace(".edus", ext)),
                                        process=lambda line: line)
                assert len(sexp) == 1
                sexp = sexp[0]
                with open(os.path.join(path_out, new_filename + ext), "w") as f:
                    f.write("%s\n" % sexp)

def main():
    config = utils.Config()
    process1(os.path.join(config.getpath("rstdt"), "TRAINING"),
             os.path.join(config.getpath("data"), "rstdt", "wsj", "train"))
    process1(os.path.join(config.getpath("rstdt"), "TEST"),
             os.path.join(config.getpath("data"), "rstdt", "wsj", "test"))

    process2(os.path.join(config.getpath("data"), "rstdt", "wsj", "train"),
             os.path.join(config.getpath("data"), "rstdt", "renamed"),
             split="train")
    process2(os.path.join(config.getpath("data"), "rstdt", "wsj", "test"),
             os.path.join(config.getpath("data"), "rstdt", "renamed"),
             split="test")

if __name__ == "__main__":
    main()
