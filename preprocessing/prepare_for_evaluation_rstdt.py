import os

import utils

import dataloader

def main():
    config = utils.Config()

    for split in ["train", "test"]:

        dataset = dataloader.read_rstdt(split=split, relation_level="coarse-grained", with_root=False)

        with open(os.path.join(config.getpath("data"), "rstdt", "wsj", split, "gold.labeled.nary.ctrees"), "w") as f:
            for data in dataset:
                f.write("%s\n" % " ".join(data.nary_sexp))

        with open(os.path.join(config.getpath("data"), "rstdt", "wsj", split, "gold.labeled.bin.ctrees"), "w") as f:
            for data in dataset:
                f.write("%s\n" % " ".join(data.bin_sexp))

        with open(os.path.join(config.getpath("data"), "rstdt", "wsj", split, "gold.arcs"), "w") as f:
            for data in dataset:
                arcs = ["%s-%s-%s" % (h,d,r) for (h,d,r) in data.arcs]
                f.write("%s\n" % " ".join(arcs))

if __name__ == "__main__":
    main()
