import argparse
import os

import pyprind

import utils
import treetk

def main(args):
    path = args.path

    filenames = os.listdir(path)
    filenames = [n for n in filenames if n.endswith(".edus.arcs")]
    filenames.sort()

    for filename in pyprind.prog_bar(filenames):
        edus_arcs = utils.read_lines(os.path.join(path, filename), process=lambda line: line.split())

        edus_deprels = []
        for arcs in edus_arcs:
            arcs = treetk.hyphens2arcs(arcs)
            deprels = [l for h,d,l in arcs]
            edus_deprels.append(deprels)

        # Write
        with open(os.path.join(path, filename.replace(".edus.arcs", ".edus.deprels")), "w") as f:
            for deprels in edus_deprels:
                deprels = " ".join(deprels)
                f.write("%s\n" % deprels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()
    main(args)
