import argparse
import os

import textpreprocessor.create_vocabulary

import utils

def main(args):
    config = utils.Config()

    utils.mkdir(os.path.join(config.getpath("data"), "rstdt-vocab"))

    filenames = os.listdir(os.path.join(config.getpath("data"), "rstdt", "renamed"))
    filenames = [n for n in filenames if n.endswith(".edus")]
    filenames.sort()

    with open(os.path.join(config.getpath("data"), "rstdt", "tmp.preprocessing", "concat.edus.heads.deprel"), "w") as f:
        for filename in filenames:
            deprels = utils.read_lines(os.path.join(config.getpath("data"), "rstdt", "renamed", filename + ".heads"),
                                     process=lambda line: line.split()[-1])
            for deprel in deprels:
                f.write("%s\n" % deprel)

    if args.with_root:
        special_words = ["<root>"]
    else:
        special_words = []
    textpreprocessor.create_vocabulary.run(
            os.path.join(config.getpath("data"), "rstdt", "tmp.preprocessing", "concat.edus.heads.deprel"),
            os.path.join(config.getpath("data"), "rstdt-vocab", "deprels.vocab.txt"),
            prune_at=10000000,
            min_count=-1,
            special_words=special_words,
            with_unk=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_root", action="store_true")
    args = parser.parse_args()
    main(args)
