from collections import defaultdict

import numpy as np
from chainer import cuda

import treetk

class IncrementalCKYDecoder(object):

    def __init__(self):
        self.decoder = CKYDecoder()

    def decode(self,
              model,
              sexps,
              edus,
              edus_postag,
              sbnds,
              pbnds,
              padded_edu_vectors,
              mask_bwd,
              mask_fwd,
              use_sbnds,
              use_pbnds,
              gold_spans=None):
        """
        :type model: Model
        :type sexps: list of int
        :type edus: list of list of str
        :type edus_postag: list of list of str
        :type sbnds: list of (int, int)
        :type pbnds: list of (int, int)
        :type padded_edu_vectors: Variable(shape=(n_edus+2, bilstm_dim+tempfeat_dim), dtype=np.float32)
        :type mask_bwd: Variable(shape=(1, bilstm_dim), dtype=np.float32)
        :type mask_fwd: Variable(shape=(1, bilstm_dim), dtype=np.float32)
        :type use_sbnds: bool
        :type use_pbnds: bool
        :type gold_spans: list of (int, int)
        :rtype: list of str
        """
        assert padded_edu_vectors.shape[0] == len(edus) + 2 # NOTE

        # Sentence-level parsing
        if use_sbnds:
            target_bnds = sbnds
            sexps = self.apply_decoder(
                                target_bnds=target_bnds,
                                model=model,
                                sexps=sexps,
                                edus=edus,
                                edus_postag=edus_postag,
                                sbnds=sbnds,
                                pbnds=pbnds,
                                padded_edu_vectors=padded_edu_vectors,
                                mask_bwd=mask_bwd,
                                mask_fwd=mask_fwd,
                                gold_spans=gold_spans)

        # Paragraph-level parsing
        if use_pbnds:
            if use_sbnds:
                target_bnds = pbnds
            else:
                target_bnds = [(sbnds[b][0],sbnds[e][1]) for b,e in pbnds]
            sexps = self.apply_decoder(
                                target_bnds=target_bnds,
                                model=model,
                                sexps=sexps,
                                edus=edus,
                                edus_postag=edus_postag,
                                sbnds=sbnds,
                                pbnds=pbnds,
                                padded_edu_vectors=padded_edu_vectors,
                                mask_bwd=mask_bwd,
                                mask_fwd=mask_fwd,
                                gold_spans=gold_spans)

        # Document-level parsing
        sexp = self.decoder.decode(
                            model=model,
                            sexps=sexps,
                            edus=edus,
                            edus_postag=edus_postag,
                            sbnds=sbnds,
                            pbnds=pbnds,
                            padded_edu_vectors=padded_edu_vectors,
                            mask_bwd=mask_bwd,
                            mask_fwd=mask_fwd,
                            gold_spans=gold_spans)
        return sexp

    def apply_decoder(self,
                     target_bnds,
                     model,
                     sexps,
                     edus,
                     edus_postag,
                     sbnds,
                     pbnds,
                     padded_edu_vectors,
                     mask_bwd,
                     mask_fwd,
                     gold_spans=None):
        """
        :type target_bnds: list of (int, int)
        :type model: Model
        :type sexps: list of int/str
        :type edus: list of list of str
        :type edus_postag: list of list of str
        :type sbnds: list of (int, int)
        :type pbnds: list of (int, int)
        :type padded_edu_vectors: Variable(shape=(n_edus+2, bilstm_dim+tempfeat_dim), dtype=np.float32)
        :type mask_bwd: Variable(shape=(1, bilstm_dim), dtype=np.float32)
        :type mask_fwd: Variable(shape=(1, bilstm_dim), dtype=np.float32)
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
                                    model=model,
                                    sexps=sexps[begin_i:end_i+1],
                                    edus=edus,
                                    edus_postag=edus_postag,
                                    sbnds=sbnds,
                                    pbnds=pbnds,
                                    padded_edu_vectors=padded_edu_vectors,
                                    mask_bwd=mask_bwd,
                                    mask_fwd=mask_fwd,
                                    gold_spans=gold_spans) # list of str
                new_sexp = " ".join(new_sexp)
            new_sexps.append(new_sexp)
        return new_sexps

class CKYDecoder(object):

    def __init__(self):
        pass

    def decode(self,
              model,
              sexps,
              edus,
              edus_postag,
              sbnds,
              pbnds,
              padded_edu_vectors,
              mask_bwd,
              mask_fwd,
              gold_spans=None):
        """
        :type model: Model
        :type sexps: list of int/str
        :type edus: list of list of str
        :type edus_postag: list of list of str
        :type sbnds: list of (int, int)
        :type pbnds: list of (int, int)
        :type padded_edu_vectors: Variable(shape=(n_edus+2, bilstm_dim+tempfeat_dim), dtype=np.float32)
        :type mask_bwd: Variable(shape=(1, bilstm_dim), dtype=np.float32)
        :type mask_fwd: Variable(shape=(1, bilstm_dim), dtype=np.float32)
        :type gold_spans: list of (int, int)
        :rtype: list of str
        """
        assert padded_edu_vectors.shape[0] == len(edus) + 2 # NOTE

        global_indices = self.get_global_indices(sexps)

        # Precompute span scores to avoid redundant calculation
        span_scores = self.precompute_span_scores(
                                model=model,
                                global_indices=global_indices,
                                edus=edus,
                                edus_postag=edus_postag,
                                sbnds=sbnds,
                                pbnds=pbnds,
                                padded_edu_vectors=padded_edu_vectors,
                                mask_bwd=mask_bwd,
                                mask_fwd=mask_fwd) # {(int, int): float}

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

                span_score = span_scores[global_i1, global_i3]
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

    def precompute_span_scores(self,
                               model,
                               global_indices,
                               edus,
                               edus_postag,
                               sbnds,
                               pbnds,
                               padded_edu_vectors,
                               mask_bwd,
                               mask_fwd):
        """
        :type model: Model
        :type global_indices: list of (int, int)
        :type edus: list of list of str
        :type edus_postag: list of list of str
        :type sbnds: list of (int, int)
        :type pbnds: list of (int, int)
        :type padded_edu_vectors: Variable(shape=(n_edus+2, bilstm_dim+tempfeat_dim), dtype=np.float32)
        :type mask_bwd: Variable(shape=(1, bilstm_dim), dtype=np.float32)
        :type mask_fwd: Variable(shape=(1, bilstm_dim), dtype=np.float32)
        :rtype: {(int, int): float}
        """
        result = {} # {(int, int): float}

        length = len(global_indices)

        spans = []
        for d in range(1, length):
            for i1 in range(0, length - d):
                i3 = i1 + d
                global_i1 = global_indices[i1][0]
                global_i3 = global_indices[i3][1]
                spans.append((global_i1, global_i3))

        span_scores = model.forward_spans_for_bracketing(
                                edus=edus,
                                edus_postag=edus_postag,
                                sbnds=sbnds,
                                pbnds=pbnds,
                                padded_edu_vectors=padded_edu_vectors,
                                mask_bwd=mask_bwd,
                                mask_fwd=mask_fwd,
                                batch_spans=[spans],
                                aggregate=False) # (1, n_spans, bilstm_dim + tempfeat_dim)
        span_scores = cuda.to_cpu(span_scores.data)[0] # (n_spans, 1)

        for span_i, (i1, i3) in enumerate(spans):
            result[i1, i3] = span_scores[span_i,0]
            result[i1, i3] += self.get_locality_bias(i1, i3) # NOTE
        return result

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

