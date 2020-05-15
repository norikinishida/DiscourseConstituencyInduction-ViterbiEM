from collections import defaultdict

import numpy as np

import treetk

class IncrementalCKYDecoder(object):

    def __init__(self):
        self.decoder = CKYDecoder()

    def decode(self,
              span_scores,
              inputs,
              sbnds,
              pbnds,
              use_sbnds,
              use_pbnds,
              gold_spans=None):
        """
        :type span_scores: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
        :type inputs: list of int
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
            inputs = self.apply_decoder(
                                span_scores=span_scores,
                                inputs=inputs,
                                target_bnds=target_bnds,
                                gold_spans=gold_spans)

        # Paragraph-level parsing
        if use_pbnds:
            if use_sbnds:
                target_bnds = pbnds
            else:
                target_bnds = [(sbnds[b][0],sbnds[e][1]) for b,e in pbnds]
            inputs = self.apply_decoder(
                                span_scores=span_scores,
                                inputs=inputs,
                                target_bnds=target_bnds,
                                gold_spans=gold_spans)

        # Document-level parsing
        sexp = self.decoder.decode(
                            span_scores=span_scores,
                            inputs=inputs,
                            gold_spans=gold_spans)
        return sexp

    def apply_decoder(self,
                     span_scores,
                     inputs,
                     target_bnds,
                     gold_spans=None):
        """
        :type span_scores: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
        :type inputs: list of int/str
        :type target_bnds: list of (int, int)
        :type gold_spans: list of (int, int)
        :rtype: list of str
        """
        outputs = [] # list of str
        for begin_i, end_i in target_bnds:
            if begin_i == end_i:
                sexp = inputs[begin_i] # int/str
                sexp = str(sexp)
            else:
                sexp = self.decoder.decode(
                                    span_scores=span_scores,
                                    inputs=inputs[begin_i:end_i+1],
                                    gold_spans=gold_spans) # list of str
                sexp = " ".join(sexp)
            outputs.append(sexp)
        return outputs

class CKYDecoder(object):

    def __init__(self):
        pass

    def decode(self,
              span_scores,
              inputs,
              gold_spans=None):
        """
        :type span_scores: numpy.ndarray(shape=(n_edus, n_edus), dtype="float")
        :type inputs: list of int/str
        :type gold_spans: list of (int, int)
        :rtype: list of str
        """
        global_indices = self.get_global_indices(inputs)

        # Initialize charts
        chart = defaultdict(float) # {(int, int): float}
        back_ptr = {} # {(int, int): int}

        length = len(inputs)

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

                span_score = span_scores[global_i1, global_i3]
                span_score += self.get_locality_bias(global_i1, global_i3) # NOTE
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

        sexp = self.recover_tree(inputs, back_ptr, 0, length-1)
        sexp = sexp.split()
        return sexp

    def get_global_indices(self, inputs):
        """
        :type inputs: list of int/str
        :rtype: list of (int, int)
        """
        global_indices = []
        for sexp in inputs:
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

    def recover_tree(self, inputs, back_ptr, i1, i3):
        """
        :type inputs: list of int/str
        :type back_ptr: {(int, int): int}
        :type i1: int
        :type i3: int
        :rtype: str
        """
        if i1 == i3:
            return "%s" % inputs[i1]
        else:
            i2 = back_ptr[i1, i3]
            inner1 = self.recover_tree(inputs, back_ptr, i1, i2)
            inner2 = self.recover_tree(inputs, back_ptr, i2+1, i3)
            return "( %s %s )" % (inner1, inner2)

    def get_locality_bias(self, i1, i3):
        """
        :type i1: int
        :type i3: int
        :rtype: float
        """
        C = 10.0
        n_edus = float(i3 - i1 + 1)
        locality_bias = C * (1.0 / n_edus)
        return locality_bias

