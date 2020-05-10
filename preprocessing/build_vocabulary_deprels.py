import os

import utils
import treetk

def main():
    config = utils.Config()

    utils.mkdir(os.path.join(config.getpath("data"), "rstdt-vocab"))

    filenames = []
    for filename in os.listdir(os.path.join(config.getpath("data"), "rstdt", "wsj", "train")):
        filenames.append(os.path.join(config.getpath("data"), "rstdt", "wsj", "train", filename))
    for filename in os.listdir(os.path.join(config.getpath("data"), "rstdt", "wsj", "test")):
        filenames.append(os.path.join(config.getpath("data"), "rstdt", "wsj", "test", filename))
    filenames = [n for n in filenames if n.endswith(".edus.arcs")]
    filenames.sort()

    tmp_path = os.path.join(config.getpath("data"), "rstdt-vocab", "tmp.txt")
    with open(tmp_path, "w") as f:
        for filename in filenames:
            for line in open(filename):
                line = line.strip().split()
                arcs = treetk.hyphens2arcs(line)
                deprels = [l for h,d,l in arcs]
                deprels = " ".join(deprels)
                f.write("%s\n" % deprels)

    utils.build_vocabulary(paths=[tmp_path],
                           path_vocab=os.path.join(config.getpath("data"), "rstdt-vocab", "deprels.vocab.txt"),
                           prune_at=50000,
                           min_count=-1,
                           special_words=["<root>"])

if __name__ == "__main__":
    main()
