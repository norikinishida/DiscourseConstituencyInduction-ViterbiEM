import os

import utils

def main():
    config = utils.Config()

    filenames = os.listdir(os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed"))
    filenames = [n for n in filenames if n.endswith(".paragraph.boundaries")]
    filenames = [n.replace(".paragraph.boundaries", ".edus") for n in filenames]
    filenames.sort()

    with open(os.path.join(
        config.getpath("data"), "ptbwsj_wo_rstdt", "tmp.preprocessing",
        "filelist.corenlp2.txt"), "w") as ff:
        for filename in filenames:
            # Path
            path_edus = os.path.join(
                            config.getpath("data"), "ptbwsj_wo_rstdt", "tmp.preprocessing",
                            filename + ".tokenized")
            path_sbnds = os.path.join(
                            config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed",
                            filename.replace(".edus", ".sentence.noproj.boundaries"))
            path_sents = os.path.join(
                            config.getpath("data"), "ptbwsj_wo_rstdt", "tmp.preprocessing",
                            filename.replace(".edus", ".sentences"))

            # Read
            edus = utils.read_lines(path_edus, process=lambda line: line.split()) # list of list of str
            sbnds = utils.read_lines(path_sbnds, process=lambda line: (int(x) for x in line.split())) # list of (int, int)

            # Create sentences based on the sentence boundaries
            sentences = []
            for begin_i, end_i in sbnds:
                sentence = edus[begin_i:end_i+1] # list of list of str
                sentence = utils.flatten_lists(sentence) # list of str
                sentences.append(sentence)

            # Write
            with open(path_sents, "w") as fs:
                for sentence in sentences:
                    fs.write("%s\n" % " ".join(sentence))
            ff.write("%s\n" % path_sents)

if __name__ == "__main__":
    main()



