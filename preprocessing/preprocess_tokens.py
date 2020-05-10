import argparse
import os
import re

import pyprind

import utils

def preprocess(token):
    token = token.lower()
    token = re.sub(r"\d", "0", token)
    return token

def main(args):
    path = args.path

    filenames = os.listdir(path)
    filenames = [n for n in filenames if n.endswith(".edus.tokens")]
    filenames.sort()

    for filename in pyprind.prog_bar(filenames):
        edus = utils.read_lines(os.path.join(path, filename), process=lambda line: line.split())

        edus = [[preprocess(token) for token in edu] for edu in edus]

        with open(os.path.join(path, filename + ".preprocessed"), "w") as f:
            for edu in edus:
                edu = " ".join(edu)
                f.write("%s\n" % edu)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    main(args)
