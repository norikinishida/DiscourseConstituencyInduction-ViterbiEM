import os

import utils

from create_sentence_boundary_rstdt import assign_edu_ids_to_sentences
from create_sentence_boundary_rstdt import adjust
from create_sentence_boundary_rstdt import compute_boundaries
from create_sentence_boundary_rstdt import test_boundaries
from create_sentence_boundary_rstdt import write_boundaries

def get_paragraph_boundaries(path_tok, path_tok2):
    """
    :type path_tok: str
    :type path_tok2: str
    :rtype: list of tuple of int

    Compute paragraph boundaries based on the tokenized file and the paragraph-splitted file
    """
    edus = read_edus(path_tok) # list of list of int
    paragraphs = read_paragraphs(path_tok2) # list of list of str

    # Assign EDU ID to each token in the paragraph list
    tokens_with_edu_ids = utils.flatten_lists(edus)
    # if len(tokens_with_edu_ids) != len(utils.flatten_lists(paragraphs)):
    #     print(path_tok, path_tok2)
    #     print(len(tokens_with_edu_ids), len(utils.flatten_lists(paragraphs)))
    assert len(tokens_with_edu_ids) == len(utils.flatten_lists(paragraphs))
    paragraphs_with_edu_ids = assign_edu_ids_to_sentences(paragraphs, tokens_with_edu_ids)

    # Adjust
    paragraphs_with_edu_ids = adjust(paragraphs_with_edu_ids, n_edus=len(edus))
    assert len(tokens_with_edu_ids) == len(utils.flatten_lists(paragraphs_with_edu_ids))

    # Compute boundaries
    bnds = compute_boundaries(paragraphs_with_edu_ids)

    # Check
    test_boundaries(bnds, n_edus=len(edus))
    return bnds

def read_edus(path):
    """
    :type path: str
    :rtype: list of list of int
    """
    edus = []

    edu_id = 0
    for line in open(path):
        line = line.strip()
        line = filter_text(line, path) # NOTE
        tokens = line.split()
        tokens = [edu_id for _ in range(len(tokens))] # NOTE
        edus.append(tokens)
        edu_id += 1

    return edus

def read_paragraphs(path):
    """
    :type path: str
    :rtype: list of list of str
    """
    paragraphs = [] # list of list of str; Each paragraph is a list of str

    # Init
    buf = []
    for line in open(path):
        line = line.strip()
        # 空行に来たら，paragraphとしてlistにappend
        if line == "":
            if len(buf) == 0:
                continue
            paragraph = " ".join(buf).strip() # str
            paragraph = filter_text(paragraph, path) # str; NOTE
            paragraph = paragraph.split() # list of str
            paragraph = ["*" for _ in range(len(paragraph))] # NOTE
            paragraphs.append(paragraph)
            # Init
            buf = []
        else:
            buf.append(line)

    if len(buf) != 0:
        paragraph = " ".join(buf).strip() # str
        paragraph = filter_text(paragraph, path) # str; NOTE
        paragraph = paragraph.split() # list of str
        paragraph = ["*" for _ in range(len(paragraph))] # NOTE
        paragraphs.append(paragraph)

    return paragraphs

def filter_text(text, path):
    """
    :type text: str
    :type path: str
    :rtype: str
    """
    text = text.replace("Inc. .", "Inc.").replace("Co. .", "Co.").strip()
    return text

def main():
    config = utils.Config()

    filenames = os.listdir(os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed"))
    filenames = [n for n in filenames if n.endswith(".edus")]
    filenames.sort()

    n_skipped = 0
    for file_i, filename in enumerate(filenames):
        path_tok = os.path.join(
                        config.getpath("data"), "ptbwsj_wo_rstdt", "tmp.preprocessing",
                        filename + ".tokenized")
        path_tok2 = os.path.join(
                        config.getpath("data"), "ptbwsj_wo_rstdt", "tmp.preprocessing",
                        filename.replace(".edus", ".txt") + ".tokenized")
        path_out = os.path.join(
                        config.getpath("data"), "ptbwsj_wo_rstdt", "tmp.preprocessing",
                        filename.replace(".edus", ".paragraph.boundaries"))

        bnds = None
        try:
            bnds = get_paragraph_boundaries(path_tok, path_tok2)
        except:
            print("Skipped %s" % path_tok)
            n_skipped += 1
        if bnds is not None:
            write_boundaries(bnds, path_out)
    print("Skipped %d files." % n_skipped)

if __name__ == "__main__":
    main()
