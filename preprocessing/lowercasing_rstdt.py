import os

import utils
import textpreprocessor.lowercase

def main():
    config = utils.Config()

    filenames = os.listdir(os.path.join(config.getpath("data"), "rstdt", "renamed"))
    filenames = [n for n in filenames if n.endswith(".edus")]
    filenames.sort()

    for filename in filenames:
        textpreprocessor.lowercase.run(
                os.path.join(
                    config.getpath("data"), "rstdt", "tmp.preprocessing",
                    filename + ".tokenized"),
                os.path.join(
                    config.getpath("data"), "rstdt", "tmp.preprocessing",
                    filename + ".tokenized.lowercased"))

if __name__ == "__main__":
    main()
