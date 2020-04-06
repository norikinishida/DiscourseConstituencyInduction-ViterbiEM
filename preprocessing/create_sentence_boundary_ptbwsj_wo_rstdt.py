import os
import utils

def get_sentence_boundaries(path_tok, path_conll):
    """
    :type path_tok: str
    :type path_conll: str
    :rtype: list of (int, int)

    Compute sentence boundaries based on the tokenized file and the sentence-splitted file
    """
    edus = read_edus(path_tok) # list of list of int
    sentences = read_sentences(path_conll) # list of list of str

    # Assign EDU ID to each token in the sentence list.
    tokens_with_edu_ids = utils.flatten_lists(edus)
    assert len(tokens_with_edu_ids) == len(utils.flatten_lists(sentences))
    sentences_with_edu_ids = assign_edu_ids_to_sentences(sentences, tokens_with_edu_ids)

    # Adjustment
    sentences_with_edu_ids = adjust(sentences_with_edu_ids, n_edus=len(edus))
    assert len(tokens_with_edu_ids) == len(utils.flatten_lists(sentences_with_edu_ids))

    # Compute boundaries
    bnds = compute_boundaries(sentences_with_edu_ids)

    # Check
    test_boundaries(bnds, n_edus=len(edus))
    return bnds

def read_edus(path):
    """
    :type path: str
    :rtype: list of list of int

    Eech EDU is a list of integer that specifies the EDU ID.
    """
    edus = []

    edu_id = 0
    for line in open(path):
        tokens = line.strip().split()
        tokens = [edu_id for _ in range(len(tokens))] # NOTE
        edus.append(tokens)
        edu_id += 1

    return edus

def read_sentences(path):
    """
    :type path: str
    :rtype: list of list of str
    """
    sentences = []

    # Init
    tokens = []
    for line in open(path):
        line = line.strip()
        if line == "":
            if len(tokens) == 0:
                continue
            sentence = ["*" for _ in range(len(tokens))] # NOTE
            sentences.append(sentence)
            # Init
            tokens = []
        else:
            items = line.split("\t")
            token = items[1]
            tokens.append(token)

    if len(tokens) != 0:
        sentence = ["*" for _ in range(len(tokens))] # NOTE
        sentences.append(sentence)
    return sentences

def assign_edu_ids_to_sentences(sentences, tokens_with_edu_ids):
    """
    :type sentences: list of list of str
    :type tokens_with_edu_ids: list of int
    :rtype: list of list of int
    """
    sentences_with_edu_ids = []
    index = 0
    for sentence in sentences:
        length = len(sentence)
        sentences_with_edu_ids.append(tokens_with_edu_ids[index:index+length])
        index += length
    return sentences_with_edu_ids

def adjust(sentences_with_edu_ids, n_edus):
    """
    :type sentences_with_edu_ids: list of list of int
    :type n_edus: int
    :rtype: list of list of int

    After using this function, each EDU belongs to only one sentence.
    e.g., [[i,i,i,i,i+1,i+1], [i+1,i+1,i+1,i+1,i+2], [i+3,i+3]]
       -> [[i,i,i,i,-1,-1], [i+1,i+1,i+1,i+1,i+2], [i+3,i+3]]
    """
    new_sentences = []

    # Record the sentence ID where the tokens in each EDU appears the most frequently
    memo = {}
    for edu_id in range(0, n_edus):
        max_count = -1
        max_sentence_id = None
        for sentence_id, sentence in enumerate(sentences_with_edu_ids):
            count = sentence.count(edu_id)
            if max_count <= count:
                max_count = count
                max_sentence_id = sentence_id
        memo[edu_id] = (max_sentence_id, max_count)

    # Replacement
    for sentence_id, sentence in enumerate(sentences_with_edu_ids):
        # Replace the token (EDU ID) with -1,
        # if this sentence is not the most-frequent sentence for the EDU (ID).
        new_sentence = [edu_id if memo[edu_id][0] == sentence_id else -1 for edu_id in sentence]
        new_sentences.append(new_sentence)
    return new_sentences

def compute_boundaries(sentences_with_edu_ids):
    """
    :type sentences_with_edu_ids: list of list of int
    :rtype: list of (int, int)
    """
    bnds = []

    for sentence in sentences_with_edu_ids:
        max_edu_id = max(sentence)
        min_edu_id = min(sentence)

        if max_edu_id == -1:
            # Empty sentence.
            continue

        if min_edu_id == -1:
            vals = set(sentence)
            vals.remove(-1)
            min_edu_id = min(vals)

        bnds.append((min_edu_id, max_edu_id))

    return bnds

def test_boundaries(bnds, n_edus):
    """
    :type bnds: list of (int, int)
    :type n_edus: int
    :rtype: bool
    """
    for edu_id in range(0, n_edus):
        check = False
        # Each EDU must belongs to at least one span.
        for begin_i, end_i in bnds:
            if begin_i <= edu_id <= end_i:
                check = True
        assert check

def write_boundaries(bnds, path):
    """
    :type bnds: list of (int, int)
    :type path: str
    """
    with open(path, "w") as f:
        for begin_i, end_i in bnds:
            f.write("%d %d\n" % (begin_i, end_i))

def main():
    config = utils.Config()

    filenames = os.listdir(os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed"))
    filenames = [n for n in filenames if n.endswith(".edus")]
    filenames.sort()

    for file_i, filename in enumerate(filenames):
        path_tok = os.path.join(
                        config.getpath("data"), "ptbwsj_wo_rstdt", "tmp.preprocessing",
                        filename + ".tokenized")
        path_conll = os.path.join(
                        config.getpath("data"), "ptbwsj_wo_rstdt", "tmp.preprocessing",
                        filename + ".tokenized.conll")
        path_out = os.path.join(
                        config.getpath("data"), "ptbwsj_wo_rstdt", "tmp.preprocessing",
                        filename.replace(".edus", ".sentence.boundaries"))

        bnds = get_sentence_boundaries(path_tok, path_conll)
        write_boundaries(bnds, path_out)

if __name__ == "__main__":
    main()

