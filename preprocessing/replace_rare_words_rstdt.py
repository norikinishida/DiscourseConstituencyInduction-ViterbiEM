import os

import pyprind

import utils
import textpreprocessor.replace_rare_words

def main():
    config = utils.Config()

    path_vocab = os.path.join(config.getpath("data"), "rstdt-vocab", "words.vocab.txt")

    filenames = os.listdir(os.path.join(config.getpath("data"), "rstdt", "renamed"))
    filenames = [n for n in filenames if n.endswith(".edus")]
    filenames.sort()

    vocab = utils.read_vocab(path_vocab)
    vocab_keys = list(vocab.keys())

    for filename in pyprind.prog_bar(filenames):
        textpreprocessor.replace_rare_words.run(
            os.path.join(
                config.getpath("data"), "rstdt", "tmp.preprocessing",
                filename + ".tokenized.lowercased.replace_digits"),
            os.path.join(
                config.getpath("data"), "rstdt", "renamed",
                filename + ".preprocessed"),
            vocab=vocab_keys)

if __name__ == "__main__":
    main()
