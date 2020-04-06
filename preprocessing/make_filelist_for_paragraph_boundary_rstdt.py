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

    filename_map = read_filename_map(os.path.join(
                        config.getpath("data"), "rstdt", "renamed",
                        "filename_map.txt"))

    with open(os.path.join(
                config.getpath("data"), "rstdt", "tmp.preprocessing",
                "filelist.ptbtokenizer2.txt"), "w") as f:
        for filename_wsj in filename_map.keys():
            filename_ren = filename_map[filename_wsj]
            split = None
            if filename_ren.startswith("train"):
                split = "TRAINING"
            elif filename_ren.startswith("test"):
                split = "TEST"
            else:
                raise ValueError("filename_ren=%s" % filename_ren)
            path_in = os.path.join(
                        config.getpath("rstdt"), split,
                        filename_wsj.replace(".edus", ""))
            path_out = os.path.join(
                        config.getpath("data"), "rstdt", "tmp.preprocessing",
                        filename_ren.replace(".edus", ".raw.tokenized"))
            f.write("%s\t%s\n" % (path_in, path_out))

if __name__ == "__main__":
    main()
