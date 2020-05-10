import argparse
import os

import utils

def main(args):
    path = args.path

    config = utils.Config()

    path_vocab = os.path.join(config.getpath("data"), "rstdt-vocab", "words.vocab.txt")

    filenames = os.listdir(path)
    filenames = [os.path.join(path, n) for n in filenames if n.endswith(".edus.tokens.preprocessed")]
    filenames.sort()

    utils.replace_oov_tokens(paths_in=filenames, paths_out=filenames, path_vocab=path_vocab)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    main(args)
