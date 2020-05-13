import os

import utils
import treetk

def main():
    config = utils.Config()

    utils.mkdir(os.path.join(config.getpath("data"), "scidtb-vocab"))

    relation_mapper = treetk.rstdt.RelationMapper(corpus_name="scidtb")

    filenames = []

    for filename in os.listdir(os.path.join(config.getpath("data"), "scidtb", "preprocessed", "train")):
        filenames.append(os.path.join(config.getpath("data"), "scidtb", "preprocessed", "train", filename))

    for filename in os.listdir(os.path.join(config.getpath("data"), "scidtb", "preprocessed", "dev", "gold")):
        filenames.append(os.path.join(config.getpath("data"), "scidtb", "preprocessed", "dev", "gold", filename))
    for filename in os.listdir(os.path.join(config.getpath("data"), "scidtb", "preprocessed", "dev", "second_annotate")):
        filenames.append(os.path.join(config.getpath("data"), "scidtb", "preprocessed", "dev", "second_annotate", filename))

    for filename in os.listdir(os.path.join(config.getpath("data"), "scidtb", "preprocessed", "test", "gold")):
        filenames.append(os.path.join(config.getpath("data"), "scidtb", "preprocessed", "test", "gold", filename))
    for filename in os.listdir(os.path.join(config.getpath("data"), "scidtb", "preprocessed", "test", "second_annotate")):
        filenames.append(os.path.join(config.getpath("data"), "scidtb", "preprocessed", "test", "second_annotate", filename))

    filenames = [n for n in filenames if n.endswith(".edus.tokens")]
    filenames.sort()

    tmp_f_path = os.path.join(config.getpath("data"), "scidtb-vocab", "tmp_f.txt")
    tmp_c_path = os.path.join(config.getpath("data"), "scidtb-vocab", "tmp_c.txt")
    with open(tmp_f_path, "w") as ff, open(tmp_c_path, "w") as fc:
        for filename in filenames:
            lines = utils.read_lines(filename.replace(".edus.tokens", ".arcs"), process=lambda line: line.split())
            assert len(lines) == 1
            line = lines[0]
            arcs = treetk.hyphens2arcs(line)
            fine_relations = [l for h,d,l in arcs]
            coarse_relations = [relation_mapper.f2c(l) for l in fine_relations]
            fine_relations = " ".join(fine_relations)
            coarse_relations = " ".join(coarse_relations)
            ff.write("%s\n" % fine_relations)
            fc.write("%s\n" % coarse_relations)

    utils.build_vocabulary(paths_file=[tmp_f_path],
                           path_vocab=os.path.join(config.getpath("data"), "scidtb-vocab", "relations.fine.vocab.txt"),
                           prune_at=50000,
                           min_count=-1,
                           special_words=["<root>"])
    utils.build_vocabulary(paths_file=[tmp_c_path],
                           path_vocab=os.path.join(config.getpath("data"), "scidtb-vocab", "relations.coarse.vocab.txt"),
                           prune_at=50000,
                           min_count=-1,
                           special_words=["<root>"])


if __name__ == "__main__":
    main()
