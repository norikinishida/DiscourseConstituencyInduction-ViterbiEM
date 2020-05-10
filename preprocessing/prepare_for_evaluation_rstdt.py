import os

import utils

import dataloader

def main():
    config = utils.Config()

    for split in ["train", "test"]:

        databatch = dataloader.read_rstdt(split=split, relation_level="coarse-grained", with_root=False)

        with open(os.path.join(config.getpath("data"), "rstdt", "wsj", split, "gold.labeled.nary.ctrees"), "w") as f:
            for sexp in databatch.batch_nary_sexp:
                f.write("%s\n" % " ".join(sexp))

        with open(os.path.join(config.getpath("data"), "rstdt", "wsj", split, "gold.labeled.bin.ctrees"), "w") as f:
            for sexp in databatch.batch_bin_sexp:
                f.write("%s\n" % " ".join(sexp))

        with open(os.path.join(config.getpath("data"), "rstdt", "wsj", split, "gold.arcs"), "w") as f:
            for arcs in databatch.batch_arcs:
                arcs = ["%s-%s-%s" % (h,d,r) for (h,d,r) in arcs]
                f.write("%s\n" % " ".join(arcs))

if __name__ == "__main__":
    main()
