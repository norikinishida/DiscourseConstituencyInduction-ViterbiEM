import os

import spacy
import pyprind

import utils


def main():
    config = utils.Config()

    path_in_root = os.path.join(config.getpath("cord19"), "pdf_json")

    filenames = os.listdir(path_in_root)
    filenames = [n for n in filenames if n.endswith(".json")]
    filenames.sort()
    n_files = len(filenames)

    path_out_root = os.path.join(config.getpath("data"), "cord19_abst")
    utils.mkdir(path_out_root)

    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "textcat"])

    cnt = 0
    for filename in pyprind.prog_bar(filenames):
        data = utils.read_json(os.path.join(path_in_root, filename))

        if not "abstract" in data:
            print("Skipped %s" % filename)
            continue

        paras = data["abstract"] # list of paragraphs

        if len(paras) != 0:
            with open(os.path.join(path_out_root, filename.replace(".json", ".doc.tokens")), "w") as f:
                for para in paras:
                    assert para["section"] == "Abstract"
                    doc = nlp(para["text"])
                    tokens = [token.text for token in doc]
                    assert len(tokens) > 0
                    tokens = " ".join(tokens)
                    f.write("%s\n" % tokens)
                    f.write("\n") # Paragraphs are separated by empty lines
            cnt += 1

    print("Processed %d/%d files." % (cnt, n_files))

if __name__ == "__main__":
    main()
