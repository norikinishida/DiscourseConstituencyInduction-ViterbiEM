import argparse
import re

import utils

def process_line(line):
    tokens = line.split()
    tokens = [process_token(token) for token in tokens]
    tokens = " ".join(tokens)
    return tokens

def process_token(token):
    token = token.lower()
    token = re.sub(r"\d", "0", token)
    return token

def main(args):
    paths_in = args.files
    paths_out = [path_in + ".preprocessed" for path_in in paths_in]
    utils.read_process_and_write(paths_in, paths_out, process=process_line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    args = parser.parse_args()
    main(args)
