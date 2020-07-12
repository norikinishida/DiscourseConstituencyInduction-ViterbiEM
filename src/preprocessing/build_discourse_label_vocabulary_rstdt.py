import os

import pyprind

import utils
import treetk

def main() :
    config = utils.Config()

    utils.mkdir(os.path.join(config.getpath("data"), "rstdt-vocab"))

    filenames = []
    for filename in os.listdir(os.path.join(config.getpath("data"), "rstdt", "wsj", "train")):
        filenames.append(os.path.join(config.getpath("data"), "rstdt", "wsj", "train", filename))
    for filename in os.listdir(os.path.join(config.getpath("data"), "rstdt", "wsj", "test")):
        filenames.append(os.path.join(config.getpath("data"), "rstdt", "wsj", "test", filename))
    filenames = [n for n in filenames if n.endswith(".labeled.bin.ctree")]
    filenames.sort()

    relation_mapper = treetk.rstdt.RelationMapper()

    frelations = []
    crelations = []
    nuclearities = []

    for filename in pyprind.prog_bar(filenames):
        sexp = utils.read_lines(filename, process=lambda line: line)
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
    main()

