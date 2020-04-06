import argparse
import os

import treetk
import textpreprocessor.create_vocabulary
import textpreprocessor.concat

import utils

def main(args):
    config = utils.Config()

    utils.mkdir(os.path.join(config.getpath("data"), "rstdt-vocab"))

    filenames = os.listdir(os.path.join(config.getpath("data"), "rstdt", "renamed"))
    filenames = [n for n in filenames if n.endswith(".edus")]
    filenames.sort()

    # Concat
    filepaths = [os.path.join(
                    config.getpath("data"), "rstdt", "tmp.preprocessing",
                    filename + ".tokenized.lowercased.replace_digits")
                    for filename in filenames]
    textpreprocessor.concat.run(
            filepaths,
            os.path.join(
                config.getpath("data"), "rstdt", "tmp.preprocessing",
                "concat.tokenized.lowercased.replace_digits"))

    # Build vocabulary
    if args.with_root:
        special_words = ["<root>"]
    else:
        special_words = []
    textpreprocessor.create_vocabulary.run(
            os.path.join(
                config.getpath("data"), "rstdt", "tmp.preprocessing",
                "concat.tokenized.lowercased.replace_digits"),
            os.path.join(
                config.getpath("data"), "rstdt-vocab",
                "words.vocab.txt"),
            prune_at=50000,
            min_count=-1,
            special_words=special_words,
            with_unk=True)

    # Build vocabulary for fine-grained/coarse-grained relations
    relation_mapper = treetk.rstdt.RelationMapper()
    frelations = []
    crelations = []
    nuclearities = []
    for filename in filenames:
        sexp = utils.read_lines(
                    os.path.join(
                        config.getpath("data"), "rstdt", "renamed",
                        filename.replace(".edus", ".labeled.bin.ctree")),
                    process=lambda line: line)
        sexp = treetk.preprocess(sexp)
        tree = treetk.rstdt.postprocess(treetk.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=False))
        nodes = treetk.traverse(tree, order="pre-order", include_terminal=False, acc=None)
        part_frelations = []
        part_crelations = []
        part_nuclearities = []
        for node in nodes:
            relations_ = node.relation_label.split("/")
            part_frelations.extend(relations_)
            part_crelations.extend([relation_mapper.f2c(r) for r in relations_])
            part_nuclearities.append(node.nuclearity_label)
        if args.with_root:
            part_frelations.append("<root>")
            part_crelations.append("<root>")
        frelations.append(part_frelations)
        crelations.append(part_crelations)
        nuclearities.append(part_nuclearities)

    fcounter = utils.get_word_counter(lines=frelations)
    ccounter = utils.get_word_counter(lines=crelations)
    ncounter = utils.get_word_counter(lines=nuclearities)
    frelations = fcounter.most_common() # list of (str, int)
    crelations = ccounter.most_common() # list of (str, int)
    nuclearities = ncounter.most_common() # list of (str, int)
    utils.write_vocab(os.path.join(config.getpath("data"), "rstdt-vocab", "relations.fine.vocab.txt"),
                      frelations)
    utils.write_vocab(os.path.join(config.getpath("data"), "rstdt-vocab", "relations.coarse.vocab.txt"),
                      crelations)
    utils.write_vocab(os.path.join(config.getpath("data"), "rstdt-vocab", "nuclearities.vocab.txt"),
                      nuclearities)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_root", action="store_true")
    args = parser.parse_args()
    main(args)

