import os

import utils

def main():
    config = utils.Config()

    filenames = os.listdir(os.path.join(config.getpath("data"), "rstdt", "renamed"))
    filenames = [n for n in filenames if n.endswith(".edus")]
    filenames.sort()

    utils.mkdir(os.path.join(config.getpath("data"), "rstdt", "tmp.preprocessing"))

    with open(os.path.join(
                config.getpath("data"), "rstdt", "tmp.preprocessing",
                "filelist.ptbtokenizer.txt"), "w") as f:
        for filename in filenames:
            path_in = os.path.join(
                            config.getpath("data"), "rstdt", "renamed",
                            filename)
            path_out = os.path.join(
                            config.getpath("data"), "rstdt", "tmp.preprocessing",
                            filename + ".tokenized")
            f.write("%s\t%s\n" % (path_in, path_out))

if __name__ == "__main__":
    main()

