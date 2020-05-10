import os

import utils

def main():
    config = utils.Config()

    path_rstdt = os.path.join(config.getpath("data"), "rstdt", "wsj")

    count = 0
    for split in ["train", "test"]:
        filenames = os.listdir(os.path.join(path_rstdt, split))
        filenames = [n for n in filenames if n.endswith(".edus.raw")]
        filenames.sort()

        for filename in filenames:
            edus = utils.read_lines(os.path.join(path_rstdt, split, filename))
            sents = utils.read_lines(os.path.join(path_rstdt, split, filename.replace(".edus.raw", ".sents.raw")))
            edus = ["".join(edu.split()) for edu in edus]
            sents = ["".join(s.split()) for s in sents]

            sbnds = []
            s_i = 0
            begin_i = 0
            buf = ""
            for end_i in range(len(edus)):
                buf += edus[end_i]
                print(filename)
                print("1-BUF:", buf)
                print("2-SNT:", sents[s_i])
                print("#############")
                if len(buf) > len(sents[s_i]):
                    sys.exit()
                    count += 1
                if buf == sents[s_i]:
                    sbnds.append((begin_i, end_i))
                    s_i += 1
                    begin_i = end_i + 1
                    buf = ""
            assert s_i == len(sents)

    print(count)

if __name__ == "__main__":
    main()
