import os

import utils

import dataloader

def main():
    config = utils.Config()

    databatch = dataloader.read_scidtb(split="train", sub_dir="")
    with open(os.path.join(config.getpath("data"), "scidtb", "preprocessed", "train", "gold.arcs"), "w") as f:
        for arcs in databatch.batch_arcs:
            arcs = ["%s-%s-%s" % (h,d,r) for (h,d,r) in arcs]
            f.write("%s\n" % " ".join(arcs))

    databatch = dataloader.read_scidtb(split="dev", sub_dir="gold")
    with open(os.path.join(config.getpath("data"), "scidtb", "preprocessed", "dev", "gold", "gold.arcs"), "w") as f:
        for arcs in databatch.batch_arcs:
            arcs = ["%s-%s-%s" % (h,d,r) for (h,d,r) in arcs]
            f.write("%s\n" % " ".join(arcs))

    databatch = dataloader.read_scidtb(split="dev", sub_dir="second_annotate")
    with open(os.path.join(config.getpath("data"), "scidtb", "preprocessed", "dev", "second_annotate", "gold.arcs"), "w") as f:
        for arcs in databatch.batch_arcs:
            arcs = ["%s-%s-%s" % (h,d,r) for (h,d,r) in arcs]
            f.write("%s\n" % " ".join(arcs))

    databatch = dataloader.read_scidtb(split="test", sub_dir="gold")
    with open(os.path.join(config.getpath("data"), "scidtb", "preprocessed", "test", "gold", "gold.arcs"), "w") as f:
        for arcs in databatch.batch_arcs:
            arcs = ["%s-%s-%s" % (h,d,r) for (h,d,r) in arcs]
            f.write("%s\n" % " ".join(arcs))

    databatch = dataloader.read_scidtb(split="test", sub_dir="second_annotate")
    with open(os.path.join(config.getpath("data"), "scidtb", "preprocessed", "test", "second_annotate", "gold.arcs"), "w") as f:
        for arcs in databatch.batch_arcs:
            arcs = ["%s-%s-%s" % (h,d,r) for (h,d,r) in arcs]
            f.write("%s\n" % " ".join(arcs))

if __name__ == "__main__":
    main()
