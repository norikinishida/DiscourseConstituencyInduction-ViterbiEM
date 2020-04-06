import os

import utils

def main():
    config = utils.Config()

    filenames = os.listdir(os.path.join(config.getpath("data"), "rstdt", "renamed"))
    filenames = [name for name in filenames if name.endswith(".edus")]
    filenames.sort()

    with open(os.path.join(
                config.getpath("data"), "rstdt", "tmp.preprocessing",
                "filelist.corenlp.txt"), "w") as f:
        for filename in filenames:
            line = os.path.join(
                        config.getpath("data"), "rstdt", "tmp.preprocessing",
                        filename + ".tokenized")
            f.write("%s\n" % line)

if __name__ == "__main__":
    main()
