import argparse
import codecs
import os

import spacy
import pyprind

import utils

def main(args):
    assert args.inside_rstdt ^ args.outside_rstdt

    config = utils.Config()

    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "textcat"])

    # Collect file names in RST-DT
    rstdt_train_filenames = os.listdir(os.path.join(config.getpath("data"), "rstdt", "wsj", "train"))
    rstdt_test_filenames = os.listdir(os.path.join(config.getpath("data"), "rstdt", "wsj", "test"))
    rstdt_train_filenames = [n for n in rstdt_train_filenames if n.endswith(".edus.tokens")]
    rstdt_test_filenames = [n for n in rstdt_test_filenames if n.endswith(".edus.tokens")]
    rstdt_train_filenames = [n[:-len(".edus.tokens")] for n in rstdt_train_filenames]
    rstdt_test_filenames = [n[:-len(".edus.tokens")] for n in rstdt_test_filenames]
    assert len(rstdt_train_filenames) == 347
    assert len(rstdt_test_filenames) == 38

    if args.outside_rstdt:
        # Prepare the target directory: /path/to/data/ptbwsj_wo_rstdt
        utils.mkdir(os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt"))

    sections = os.listdir(config.getpath("ptbwsj"))
    sections.sort()
    count = 0
    for section in pyprind.prog_bar(sections):
        # File names of articles in PTB-WSJ
        filenames = os.listdir(os.path.join(config.getpath("ptbwsj"), section))
        filenames = [n for n in filenames if n.startswith("wsj_")]
        filenames.sort()

        for filename in filenames:
            # Read text of the article
            try:
                lines = utils.read_lines(
                                os.path.join(config.getpath("ptbwsj"), section, filename),
                                process=lambda line: line)
            except UnicodeDecodeError:
                lines = []
                for line in codecs.open(os.path.join(config.getpath("ptbwsj"), section, filename), "r", "latin-1"):
                    line = line.strip()
                    lines.append(line)

            # Remove the ".START" markers
            assert lines[0] == ".START"
            lines = lines[1:]
            for i in range(len(lines)):
                lines[i] = lines[i].replace(".START", "")
                lines[i] = " ".join(lines[i].split())

            # Remove the beginning empty lines
            top_empty_count = 0
            for line_i in range(len(lines)):
                if lines[line_i] == "":
                    top_empty_count += 1
                else:
                    break
            lines = lines[top_empty_count:]

            # Tokenization
            tokenized_lines = []
            for line in lines:
                if line == "":
                    tokens = ""
                else:
                    doc = nlp(line)
                    tokens = [token.text for token in doc]
                    tokens = " ".join(tokens)
                tokenized_lines.append(tokens)

            if args.inside_rstdt:
                if filename in rstdt_train_filenames:
                    # File inside RST-DT training set
                    utils.write_lines(
                            os.path.join(config.getpath("data"), "rstdt", "wsj", "train", filename + ".doc.tokens"),
                            tokenized_lines)
                    count += 1
                elif filename in rstdt_test_filenames:
                    # File inside RST-DT test set
                    utils.write_lines(
                            os.path.join(config.getpath("data"), "rstdt", "wsj", "test", filename + ".doc.tokens"),
                            tokenized_lines)
                    count += 1
                else:
                    continue
            else:
                if filename in rstdt_train_filenames:
                    continue
                elif filename in rstdt_test_filenames:
                    continue
                else:
                    # File outside RST-DT
                    utils.write_lines(
                            os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", filename + ".doc.tokens"),
                            tokenized_lines)
                    count += 1

    print("Processed %d files." % count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inside_rstdt", action="store_true")
    parser.add_argument("--outside_rstdt", action="store_true")
    args = parser.parse_args()
    main(args)

