import os

import pyprind

import utils

def remove_empty_lines(filename, lines):
    """
    :type filename: str
    :type lines: list of str
    :rtype: list of str
    """
    n_lines = len(lines)
    lines = [line for line in lines if line != ""]
    if len(lines) != n_lines:
        # print("Removed %d empty lines in %s." % (n_lines - len(lines), filename))
        pass
    return lines

def count_chars(lines):
    """
    :type lines: list of str
    :rtype: int
    """
    n_chars = 0
    for line in lines:
        n_chars += len([c for c in list(line) if c != " "])
    return n_chars

def convert_edus(edus, raw_lines):
    """
    :type edus: list of str
    :type raw_lines: list of str
    :rtype: list of str
    """
    edu_positions = []
    for edu_i in range(len(edus)):
        raw = []
        for char_i in range(len(list(edus[edu_i]))):
            if edus[edu_i][char_i] == " ":
                continue
            raw.append(edu_i)
        edu_positions.append(raw)
    edu_positions = utils.flatten_lists(edu_positions)
    # print(edu_positions)

    flatten_raw_lines = list("".join(utils.flatten_lists(raw_lines)))
    # print(flatten_raw_lines)
    result_positions = [-1 for _ in flatten_raw_lines]
    result_i = 0
    cur_char_i = 0
    cur_edu_i = 0
    for char in flatten_raw_lines:
        if char == " ":
            result_positions[result_i] = cur_edu_i
        else:
            edu_i = edu_positions[cur_char_i]
            result_positions[result_i] = edu_i
            cur_char_i += 1
            assert edu_i == cur_edu_i or edu_i == cur_edu_i + 1
            cur_edu_i = edu_i
        result_i += 1

    # print(result_positions)

    new_edus = []
    for edu_i in range(len(edus)):
        b = result_positions.index(edu_i)
        e = b + result_positions.count(edu_i)
        new_edu = "".join(flatten_raw_lines[b:e])
        new_edu = new_edu.strip()
        # print(new_edu)
        new_edus.append(new_edu)

    return new_edus

def main():
    config = utils.Config()

    filenames = os.listdir(os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "segmented"))
    filenames = [n for n in filenames if n.endswith(".txt")]
    filenames.sort()

    utils.mkdir(os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed"))

    for filename in pyprind.prog_bar(filenames):
        path_seg = os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "segmented", filename)
        path_raw = os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "raw", filename)
        path_dst = os.path.join(config.getpath("data"), "ptbwsj_wo_rstdt", "preprocessed", filename.replace(".txt", ".edus"))
        # Input
        edus = utils.read_lines(path_seg, process=lambda line: line)
        edus = remove_empty_lines(filename, edus)
        raw_lines = utils.read_lines(path_raw, process=lambda line: line)
        raw_lines = remove_empty_lines(filename, raw_lines)
        assert count_chars(edus) == count_chars(raw_lines)
        # Processing
        edus = convert_edus(edus, raw_lines)
        assert count_chars(edus) == count_chars(raw_lines)
        # Output
        utils.write_lines(path_dst, edus)

if __name__ == "__main__":
    main()
