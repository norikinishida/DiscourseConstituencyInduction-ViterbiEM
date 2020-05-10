import os

import utils

def main():
    config = utils.Config()

    utils.mkdir(os.path.join(config.getpath("data"), "rstdt-vocab"))

    filenames = []
    for filename in os.listdir(os.path.join(config.getpath("data"), "rstdt", "wsj", "train")):
        filenames.append(os.path.join(config.getpath("data"), "rstdt", "wsj", "train", filename))
    for filename in os.listdir(os.path.join(config.getpath("data"), "rstdt", "wsj", "test")):
        filenames.append(os.path.join(config.getpath("data"), "rstdt", "wsj", "test", filename))
    filenames = [n for n in filenames if n.endswith(".edus.postags")]
    filenames.sort()

    utils.build_vocabulary(paths=filenames,
                           path_vocab=os.path.join(config.getpath("data"), "rstdt-vocab", "postags.vocab.txt"),
                           prune_at=50000,
                           min_count=-1,
                           special_words=["<root>"])

if __name__ == "__main__":
    main()
