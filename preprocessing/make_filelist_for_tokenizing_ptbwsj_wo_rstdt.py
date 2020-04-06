import os

import utils

def read_filename_map(path):
    """
    :type path: str
    :rtype: {str -> str}
    """
    lines = utils.read_lines(path, process=lambda line: [os.path.basename(item) for item in line.split()])
    filename_map = {key:val for key,val in lines}
    return filename_map

def main():
    config = utils.Config()

    filenames = os.listdir(os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed"))
    filenames = [n for n in filenames if n.endswith(".edus")]
    filenames.sort()

    utils.mkdir(os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "tmp.preprocessing"))

    with open(os.path.join(
                config.getpath("data"),
                "ptbwsj_wo_rstdt",
                "tmp.preprocessing",
                "filelist.ptbtokenizer.txt"), "w") as f:
        for filename in filenames:
            path_in = os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed", filename)
            path_out = os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "tmp.preprocessing", filename + ".tokenized")
            f.write("%s\t%s\n" % (path_in, path_out))

if __name__ == "__main__":
    main()
