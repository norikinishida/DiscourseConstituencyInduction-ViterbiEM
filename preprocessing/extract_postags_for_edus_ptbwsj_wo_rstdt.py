import os

import utils

def main():
    config = utils.Config()

    filenames = os.listdir(os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed"))
    filenames = [n for n in filenames if n.endswith(".paragraph.boundaries")]
    filenames = [n.replace(".paragraph.boundaries", ".edus") for n in filenames]
    filenames.sort()

    for filename in filenames:
        # Path
        path_edus = os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "tmp.preprocessing", filename + ".tokenized")
        path_conll = os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "tmp.preprocessing", filename.replace(".edus", ".sentences.conll"))
        path_out = os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed", filename + ".postags")

        # Read
        edus = utils.read_lines(path_edus, process=lambda line: line.split()) # list of list of str
        tokens_e = utils.flatten_lists(edus) # list of str

        sentences = utils.read_conll(path_conll, keys=["ID", "FORM", "LEMMA", "POSTAG", "_1", "HEAD", "DEPREL"]) # list of list of {str: str}
        conll_lines = utils.flatten_lists(sentences) # list of {str: str}
        tokens_s = [conll_line["FORM"] for conll_line in conll_lines] # list of str
        postags_s = [conll_line["POSTAG"] for conll_line in conll_lines] # list of str

        # Check whether the number of tokens and that of postags are equivalent
        for token_e, token_s, postag_s in zip(tokens_e, tokens_s, postags_s):
            if token_e != token_s:
                raise ValueError("Error! %s != %s" % (token_e, token_s))

        # Create the POSTAG-version of EDUs
        postag_i = 0
        edus_postag = []
        for edu in edus:
            edu_postag = [postags_s[postag_i + i] for i in range(len(edu))]
            edus_postag.append(edu_postag)
            postag_i += len(edu)

        # Write
        with open(path_out, "w") as f:
            for edu_postag in edus_postag:
                f.write("%s\n" % " ".join(edu_postag))

if __name__ == "__main__":
    main()
