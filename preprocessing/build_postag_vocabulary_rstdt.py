import argparse
import os

import textpreprocessor.create_vocabulary
import textpreprocessor.concat

import utils

def main(args):
    config = utils.Config()

    utils.mkdir(os.path.join(config.getpath("data"), "rstdt-vocab"))

    filenames = os.listdir(os.path.join(config.getpath("data"), "rstdt", "renamed"))
    filenames = [n for n in filenames if n.endswith(".edus")]
    filenames.sort()

    filepaths = [os.path.join(config.getpath("data"), "rstdt", "renamed", filename + ".postags")
                 for filename in filenames]

    # Concat
    textpreprocessor.concat.run(
            filepaths,
            os.path.join(config.getpath("data"), "rstdt", "tmp.preprocessing", "concat.edus.postags"))

    # Build vocabulary
    if args.with_root:
        special_words = ["<root>"]
    else:
        special_words = []
    textpreprocessor.create_vocabulary.run(
            os.path.join(config.getpath("data"), "rstdt", "tmp.preprocessing", "concat.edus.postags"),
            os.path.join(config.getpath("data"), "rstdt-vocab", "postags.vocab.txt"),
            prune_at=50000,
            min_count=-1,
            special_words=special_words,
            with_unk=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_root", action="store_true")
    args = parser.parse_args()
    main(args)
