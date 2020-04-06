import os

import utils

import dataloader

def main():
    config = utils.Config()

    train_databatch = dataloader.read_rstdt(split="train", relation_level="coarse-grained", with_root=False)
    test_databatch = dataloader.read_rstdt(split="test", relation_level="coarse-grained", with_root=False)
    all_databatch = utils.concat_databatch(train_databatch, test_databatch)

    with open(os.path.join(config.getpath("data"), "rstdt", "renamed", "train.labeled.nary.ctrees"), "w") as f:
        for sexp in train_databatch.batch_nary_sexp:
            f.write("%s\n" % " ".join(sexp))

    with open(os.path.join(config.getpath("data"), "rstdt", "renamed", "train.labeled.bin.ctrees"), "w") as f:
        for sexp in train_databatch.batch_bin_sexp:
            f.write("%s\n" % " ".join(sexp))

    with open(os.path.join(config.getpath("data"), "rstdt", "renamed", "test.labeled.nary.ctrees"), "w") as f:
        for sexp in test_databatch.batch_nary_sexp:
            f.write("%s\n" % " ".join(sexp))

    with open(os.path.join(config.getpath("data"), "rstdt", "renamed", "test.labeled.bin.ctrees"), "w") as f:
        for sexp in test_databatch.batch_bin_sexp:
            f.write("%s\n" % " ".join(sexp))

    with open(os.path.join(config.getpath("data"), "rstdt", "renamed", "all.labeled.nary.ctrees"), "w") as f:
        for sexp in all_databatch.batch_nary_sexp:
            f.write("%s\n" % " ".join(sexp))

    with open(os.path.join(config.getpath("data"), "rstdt", "renamed", "all.labeled.bin.ctrees"), "w") as f:
        for sexp in all_databatch.batch_bin_sexp:
            f.write("%s\n" % " ".join(sexp))

if __name__ == "__main__":
    main()
