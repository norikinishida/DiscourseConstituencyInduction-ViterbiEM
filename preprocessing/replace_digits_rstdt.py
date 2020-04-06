import os

import utils
import textpreprocessor.replace_digits

def main():
    config = utils.Config()

    filenames = os.listdir(os.path.join(config.getpath("data"), "rstdt", "renamed"))
    filenames = [n for n in filenames if n.endswith(".edus")]
    filenames.sort()

    for filename in filenames:
        textpreprocessor.replace_digits.run(
                os.path.join(
                    config.getpath("data"), "rstdt", "tmp.preprocessing",
                    filename + ".tokenized.lowercased"),
                os.path.join(
                    config.getpath("data"), "rstdt", "tmp.preprocessing",
                    filename + ".tokenized.lowercased.replace_digits"))

if __name__ == "__main__":
    main()
