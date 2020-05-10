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
        # EDUs (sub-arcs, tokens, POS tags)
        edus_arcs = utils.read_lines(os.path.join(path, filename), process=lambda line: line.split())
        edus_tokens = utils.read_lines(os.path.join(path, filename.replace(".edus.arcs", ".edus.tokens")), process=lambda line: line.split())
        edus_postags = utils.read_lines(os.path.join(path, filename.replace(".edus.arcs", ".edus.postags")), process=lambda line: line.split())

        heads = []
        for tokens, postags, arcs in zip(edus_tokens, edus_postags, edus_arcs):
            arcs = treetk.hyphens2arcs(arcs)

            # Check: Arcs should be arranged in ascending order wrt dependent
            prev_d = -1
            for h,d,l in arcs:
                assert d > prev_d
                prev_d = d

            head_idx = None

            # If its head is the root, it is the head of the EDU
            for idx, (h,d,l) in enumerate(arcs):
                if h == 0:
                    assert l == "ROOT" # TODO
                    head_idx = idx
                    break

            # If its head is outside the span, it is the head of the EDU
            if head_idx is None:
                span_min = arcs[0][1]
                span_max = arcs[-1][1]
                for idx, (h,d,l) in enumerate(arcs):
                    if h < span_min or h > span_max:
                        head_idx = idx
                        break

            # Head token, POS tag, and dependency relation
            head_token = tokens[head_idx]
            head_postag = postags[head_idx]
            head_deprel = arcs[head_idx][2]
            heads.append((head_token, head_postag, head_deprel))

        # Write
        with open(os.path.join(path, filename.replace(".edus.arcs", ".edus.heads")), "w") as f:
            for token, postag, deprel in heads:
                f.write("%s %s %s\n" % (token, postag, deprel))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()
    main(args)
