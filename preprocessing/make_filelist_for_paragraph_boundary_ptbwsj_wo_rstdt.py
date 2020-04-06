import os

import utils

def main():
    config = utils.Config()

    filenames = os.listdir(os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed"))
    filenames = [n for n in filenames if n.endswith(".edus")]
    filenames.sort()

    with open(os.path.join(
                config.getpath("data"), "ptbwsj_wo_rstdt", "tmp.preprocessing",
                "filelist.ptbtokenizer2.txt"), "w") as f:
        for filename in filenames:
            path_in = os.path.join(
                        config.getpath("data"), "ptbwsj_wo_rstdt", "raw",
                        filename.replace(".edus", ".txt"))
            path_out = os.path.join(
                        config.getpath("data"), "ptbwsj_wo_rstdt", "tmp.preprocessing",
                        filename.replace(".edus", ".txt") + ".tokenized")
            f.write("%s\t%s\n" % (path_in, path_out))

if __name__ == "__main__":
    main()
