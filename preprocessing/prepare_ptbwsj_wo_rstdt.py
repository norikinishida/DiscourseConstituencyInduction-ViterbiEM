import codecs
import os

import utils

def get_rstdt_wsj_filenames():
    config = utils.Config()

    filenames = []
    for split in ["train", "test"]:
        filenames_ = os.listdir(os.path.join(config.getpath("data"), "rstdt", "wsj", split))
        filenames_ = [n for n in filenames_ if n.endswith(".edus")]
        filenames.extend(filenames_)

    assert len(filenames) == 385

    filenames.remove("file1.edus")
    filenames.remove("file2.edus")
    filenames.remove("file3.edus")
    filenames.remove("file4.edus")
    filenames.remove("file5.edus")

    filenames = [n[:-len(".out.edus")] for n in filenames]

    filenames.append("wsj_0764") # "file1"
    filenames.append("wsj_0430") # "file2"
    filenames.append("wsj_0766") # "file3"
    filenames.append("wsj_0778") # "file4"
    filenames.append("wsj_2172") # "file5"

    return filenames

def main():
    config = utils.Config()

    path_out = os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt")
    utils.mkdir(path_out)
    utils.mkdir(os.path.join(path_out, "raw"))

    sections = os.listdir(config.getpath("ptbwsj"))
    sections.sort()
    rstdt_wsj_filenames = get_rstdt_wsj_filenames()
    count = 0
    for sec_i, section in enumerate(sections):
        print("[%d/%d] Processing %s" % \
                (sec_i+1, len(sections),
                 os.path.join(config.getpath("ptbwsj"), section)))

        filenames = os.listdir(os.path.join(config.getpath("ptbwsj"), section))
        filenames = [n for n in filenames if n.startswith("wsj_")]
        filenames.sort()
        for filename in filenames:
            if filename in rstdt_wsj_filenames:
                print("Skipped %s (which is contained in RST-DT)" % filename)
                continue
            count += 1

            try:
                lines = utils.read_lines(
                                os.path.join(config.getpath("ptbwsj"), section, filename),
                                process=lambda line: line)
            except UnicodeDecodeError:
                lines = []
                for line in codecs.open(os.path.join(config.getpath("ptbwsj"), section, filename), "r", "latin-1"):
                    line = line.strip()
                    lines.append(line)
            assert lines[0] == ".START"
            lines = lines[1:]
            top_empty_count = 0
            for line_i in range(len(lines)):
                if lines[line_i] == "":
                    top_empty_count += 1
                else:
                    break
            lines = lines[top_empty_count:]
            utils.write_lines(
                    os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "raw", filename + ".txt"),
                    lines)

    print("Processed %d files." % count)

if __name__ == "__main__":
    main()
