import os

import spacy
import pyprind

import utils

def extract_abstract(path):
    lines = utils.read_lines(path, process=lambda line: line)
    abst_lines = []
    in_abstract = False
    for line in lines:
        if line.lower().startswith("abstract"):
            in_abstract = True
            # Remove the beggining token(="Abstract", "ABSTRACT", etc.)
            tokens = line.split()
            tokens = tokens[1:]
            if len(tokens) > 0:
                line = " ".join(tokens)
                abst_lines.append(line)
        elif "introduction" in line.lower() or line.startswith("1"):
            in_abstract = False
            break
        elif in_abstract:
            abst_lines.append(line)
    while True:
        length = len(abst_lines)
        for i in range(len(abst_lines)):
            if abst_lines[i].endswith("-") and len(abst_lines[i]) != 1 and abst_lines[i][-2] != "-":
                if i+1 < len(abst_lines):
                    line = abst_lines[i]
                    line = line[:-1]
                    abst_lines[i+1] = line + abst_lines[i+1]
                    abst_lines.pop(i)
                    break
        if length == len(abst_lines):
            break
    abst_text = " ".join(abst_lines)
    return abst_text

def main():
    config = utils.Config()

    path_out = os.path.join(config.getpath("data"), "aarc_abst")
    utils.mkdir(path_out)

    filenames = os.listdir(config.getpath("aarc"))
    filenames = [n for n in filenames if n.endswith(".txt.utf8")]
    filenames.sort()

    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "textcat"])

    cnt = 0
    for filename in pyprind.prog_bar(filenames):
        text = extract_abstract(os.path.join(config.getpath("aarc"), filename))
        if text == "":
            # print("No Abstract!: %s" % filename)
            continue
        with open(os.path.join(path_out, filename.replace(".txt.utf8", ".doc.tokens")), "w") as f:
            doc = nlp(text)
            tokens = [token.text for token in doc]
            assert len(tokens) > 0
            tokens = " ".join(tokens)
            f.write("%s\n" % tokens)
        cnt += 1

    print("Processed %d/%d files" % (cnt, len(filenames)))

if __name__ == "__main__":
    main()
