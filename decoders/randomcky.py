from collections import defaultdict

import numpy as np

import treetk

class IncrementalRandomCKYDecoder(object):

    def __init__(self):
        self.decoder = RandomCKYDecoder()

    def decode(self,
              sexps,
              sbnds,
              pbnds,
              use_sbnds,
              use_pbnds,
              gold_spans=None):
        """
        :type sexps: list of int
        :type sbnds: list of (int, int)
        :type pbnds: list of (int, int)
        :type use_sbnds: bool
        :type use_pbnds: bool
        :type gold_spans: list of (int, int)
        :rtype: list of str
        """
        # Sentence-level parsing
        if use_sbnds:
            target_bnds = sbnds
            sexps = self.apply_decoder(
                                sexps=sexps,
                                target_bnds=target_bnds,
                                sbnds=sbnds,
                                pbnds=pbnds,
                                gold_spans=gold_spans)

        # Paragraph-level parsing
        if use_pbnds:
            if use_sbnds:
                target_bnds = pbnds
            else:
                target_bnds = [(sbnds[b][0],sbnds[e][1]) for b,e in pbnds]
            sexps = self.apply_decoder(
                                sexps=sexps,
                                target_bnds=target_bnds,
                                sbnds=sbnds,
                                pbnds=pbnds,
                                gold_spans=gold_spans)

        # Document-level parsing
        sexp = self.decoder.decode(
                            sexps=sexps,
                            gold_spans=gold_spans)
        return sexp

    def apply_decoder(self,
                     sexps,
                     target_bnds,
                     sbnds,
                     pbnds,
                     gold_spans=None):
        """
        :type sexps: list of int/str
        :type target_bnds: list of (int, int)
        :type sbnds: list of (int, int)
        :type pbnds: list of (int, int)
        :type gold_spans: list of (int, int)
        :rtype: list of str
        """
        new_sexps = [] # list of str
        for begin_i, end_i in target_bnds:
            if begin_i == end_i:
                new_sexp = sexps[begin_i] # int/str
                new_sexp = str(new_sexp)
            else:
                new_sexp = self.decoder.decode(
                                    sexps=sexps[begin_i:end_i+1],
                                    gold_spans=gold_spans) # list of str
                new_sexp = " ".join(new_sexp)
            new_sexps.append(new_sexp)
        return new_sexps

class RandomCKYDecoder(object):

    def __init__(self):
        pass

    def decode(self,
              sexps,
              gold_spans=None):
        """
        :type sexps: list of int/str
        :type gold_spans: list of (int, int)
        :rtype: list of str
        """
        global_indices = self.get_global_indices(sexps)

        # Initialize charts
        chart = defaultdict(float) # {(int, int): float}
        back_ptr = {} # {(int, int): int}

        length = len(sexps)

        # Base case
        for i in range(length):
            chart[i, i] = 1.0
            back_ptr[i, i] = None

        # General case
        for d in range(1, length):
            for i1 in range(0, length- d):
                i3 = i1 + d
                global_i1 = global_indices[i1][0]
                global_i3 = global_indices[i3][1]

                span_score = np.random.random()
                if gold_spans is not None:
                    if not (global_i1, global_i3) in gold_spans:
                        span_score += 1.0

                max_score = -np.inf
                memo = None
                for i2 in range(i1, i3):
                    score = span_score + chart[i1, i2] + chart[i2+1, i3]
                    if score > max_score:
                        max_score = score
                        memo = i2
                chart[i1, i3] = max_score
                back_ptr[i1, i3] = memo

        sexp = self.recover_tree(sexps, back_ptr, 0, length-1)
        sexp = sexp.split()
        return sexp

    def get_global_indices(self, sexps):
        """
        :type sexps: list of int/str
        :rtype: list of (int, int)
        """
        global_indices = []
        for sexp in sexps:
            if isinstance(sexp, int):
                begin_i = sexp
                end_i = sexp
            else:
                indices = treetk.filter_parens(treetk.preprocess(sexp), LPAREN="(", RPAREN=")")
                indices = [int(index) for index in indices]
                begin_i = min(indices)
                end_i = max(indices)
            global_indices.append((begin_i, end_i))
        return global_indices

    def recover_tree(self, sexps, back_ptr, i1, i3):
        """
        :type sexps: list of int/str
        :type back_ptr: {(int, int): int}
        :type i1: int
        :type i3: int
        :rtype: str
        """
        if i1 == i3:
            return "%s" % sexps[i1]
        else:
            i2 = back_ptr[i1, i3]
            inner1 = self.recover_tree(sexps, back_ptr, i1, i2)
            inner2 = self.recover_tree(sexps, back_ptr, i2+1, i3)
            return "( %s %s )" % (inner1, inner2)

