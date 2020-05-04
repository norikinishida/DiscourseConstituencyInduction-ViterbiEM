from collections import Counter

import numpy as np
import pyprind

import utils
from utils import TemplateFeatureExtractor
import treetk

class TemplateFeatureExtractor(TemplateFeatureExtractor):

    def __init__(self, databatch):
        """
        :type databatch: DataBatch
        """
        super().__init__()
        self.LENGTH_TOKENS_SPANS = [(1,2), (3,5), (6,10), (11,20), (21,np.inf)]
        self.LENGTH_EDUS_SPANS = [(1,1), (2,2), (3,3), (4,7), (8,11), (12,np.inf)]
        self.LENGTH_SENTS_SPANS = [(0,0), (1,1), (2,2), (3,5), (6,np.inf)]
        self.LENGTH_PARAS_SPANS = [(0,0), (1,1), (2,2), (3,5), (6,np.inf)]
        self.DISTANCE_SPANS = [(0,0), (1,2), (3,5), (6,np.inf)]
        self.aggregate_templates(databatch=databatch)
        self.prepare()

    ############################
    def build_ngrams(self, batch_edus, batch_nary_sexp, threshold):
        """
        :type batch_edus: list of list of list of str
        :type batch_nary_sexp: list of list of str
        :type threshold: int
        :rtype: list of (str, int), list of (str, int), list of (str, int), list of (str, int), list of (str, int), list of (str, int)
        """
        # Counting
        counter_span_begin = Counter()
        counter_span_end = Counter()
        counter_lc_begin = Counter()
        counter_lc_end = Counter()
        counter_rc_begin = Counter()
        counter_rc_end = Counter()
        prog_bar = pyprind.ProgBar(len(batch_edus))
        for edus, sexp in zip(batch_edus, batch_nary_sexp):
            ngrams_span_begin = []
            ngrams_span_end = []
            ngrams_lc_begin = []
            ngrams_lc_end = []
            ngrams_rc_begin = []
            ngrams_rc_end = []
            # Aggregate spans from the constituent tree
            tree = treetk.rstdt.postprocess(treetk.sexp2tree(sexp, with_nonterminal_labels=True, with_terminal_labels=False))
            tree.calc_spans()
            spans = treetk.aggregate_spans(tree, include_terminal=False, order="pre-order") # list of (int, int, str)
            n_edus = len(edus)
            for span in spans:
                b, e, l = span
                # Extract N-grams from a span
                part_ngrams_span_begin = self.extract_ngrams(edus[b], position="begin") # list of str
                part_ngrams_span_end = self.extract_ngrams(edus[e], position="end")
                ngrams_span_begin.extend(part_ngrams_span_begin)
                ngrams_span_end.extend(part_ngrams_span_end)
                # Extract N-grams from the left-context EDU
                if b > 0:
                    part_ngrams_lc_begin = self.extract_ngrams(edus[b-1], position="begin")
                    part_ngrams_lc_end = self.extract_ngrams(edus[b-1], position="end")
                    ngrams_lc_begin.extend(part_ngrams_lc_begin)
                    ngrams_lc_end.extend(part_ngrams_lc_end)
                # Extract N-grams from the right-context EDU
                if e < n_edus-1:
                    part_ngrams_rc_begin = self.extract_ngrams(edus[e+1], position="begin")
                    part_ngrams_rc_end = self.extract_ngrams(edus[e+1], position="end")
                    ngrams_rc_begin.extend(part_ngrams_rc_begin)
                    ngrams_rc_end.extend(part_ngrams_rc_end)
            counter_span_begin.update(ngrams_span_begin)
            counter_span_end.update(ngrams_span_end)
            counter_lc_begin.update(ngrams_lc_begin)
            counter_lc_end.update(ngrams_lc_end)
            counter_rc_begin.update(ngrams_rc_begin)
            counter_rc_end.update(ngrams_rc_end)
            prog_bar.update()

        # Filtering
        counter_span_begin = [(ngram,cnt) for ngram,cnt in counter_span_begin.most_common() if cnt >= threshold]
        counter_span_end = [(ngram,cnt) for ngram,cnt in counter_span_end.most_common() if cnt >= threshold]
        counter_lc_begin = [(ngram,cnt) for ngram,cnt in counter_lc_begin.most_common() if cnt >= threshold]
        counter_lc_end = [(ngram,cnt) for ngram,cnt in counter_lc_end.most_common() if cnt >= threshold]
        counter_rc_begin = [(ngram,cnt) for ngram,cnt in counter_rc_begin.most_common() if cnt >= threshold]
        counter_rc_end = [(ngram,cnt) for ngram,cnt in counter_rc_end.most_common() if cnt >= threshold]

        return counter_span_begin, counter_span_end,\
                counter_lc_begin, counter_lc_end,\
                counter_rc_begin, counter_rc_end

    def extract_ngrams(self, edu, position):
        """
        :type edu: list of str
        :type position: str
        :rtype: list of str
        """
        N = 2
        ngrams = []
        if position == "begin":
            for i in range(0, min(N, len(edu))):
                ngrams.append(" ".join(edu[:i+1]))
        elif position == "end":
            for i in range(0, min(N, len(edu))):
                ngrams.append(" ".join(edu[-1-i:]))
        else:
            raise ValueError("Invalid position=%s" % position)

        return ngrams
    ############################

    ############################
    def aggregate_templates(self, databatch):
        """
        :type databatch: DataBatch
        :rtype: None
        """
        ###################
        # lex_ngrams_span_begin, lex_ngrams_span_end,\
        #     lex_ngrams_lc_begin, lex_ngrams_lc_end,\
        #     lex_ngrams_rc_begin, lex_ngrams_rc_end \
        #             = self.build_ngrams(batch_edus=databatch.batch_edus,
        #                                 batch_nary_sexp=databatch.batch_nary_sexp,
        #                                 threshold=5)
        # pos_ngrams_span_begin, pos_ngrams_span_end,\
        #     pos_ngrams_lc_begin, pos_ngrams_lc_end,\
        #     pos_ngrams_rc_begin, pos_ngrams_rc_end \
        #             = self.build_ngrams(batch_edus=databatch.batch_edus_postag,
        #                                 batch_nary_sexp=databatch.batch_nary_sexp,
        #                                 threshold=5)
        ###################

        ###################
        # Features on N-grams
        # for ngram, cnt in lex_ngrams_span_begin:
        #     self.add_template(lex_ngram_span_begin=ngram)
        # for ngram, cnt in lex_ngrams_span_end:
        #     self.add_template(lex_ngram_span_end=ngram)
        # for ngram, cnt in lex_ngrams_lc_begin:
        #     self.add_template(lex_ngram_left_context_begin=ngram)
        # for ngram, cnt in lex_ngrams_lc_end:
        #     self.add_template(lex_ngram_left_context_end=ngram)
        # for ngram, cnt in lex_ngrams_lc_begin:
        #     self.add_template(lex_ngram_right_context_begin=ngram)
        # for ngram, cnt in lex_ngrams_lc_end:
        #     self.add_template(lex_ngram_right_context_end=ngram)
        #
        # for ngram, cnt in pos_ngrams_span_begin:
        #     self.add_template(pos_ngram_span_begin=ngram)
        # for ngram, cnt in pos_ngrams_span_end:
        #     self.add_template(pos_ngram_span_end=ngram)
        # for ngram, cnt in pos_ngrams_lc_begin:
        #     self.add_template(pos_ngram_left_context_begin=ngram)
        # for ngram, cnt in pos_ngrams_lc_end:
        #     self.add_template(pos_ngram_left_context_end=ngram)
        # for ngram, cnt in pos_ngrams_lc_begin:
        #     self.add_template(pos_ngram_right_context_begin=ngram)
        # for ngram, cnt in pos_ngrams_lc_end:
        #     self.add_template(pos_ngram_right_context_end=ngram)
        ###################

        # Features on size
        for span in self.LENGTH_TOKENS_SPANS:
            self.add_template(length_tokens="%s~%s" % span)
        for span in self.LENGTH_EDUS_SPANS:
            self.add_template(length_edus="%s~%s" % span)
        for span in self.LENGTH_SENTS_SPANS:
            self.add_template(length_sentences="%s~%s" % span)
        for span in self.LENGTH_PARAS_SPANS:
            self.add_template(length_paragraphs="%s~%s" % span)

        # Features on position
        for span in self.DISTANCE_SPANS:
            self.add_template(dist_from_begin="%s~%s" % span)
        for span in self.DISTANCE_SPANS:
            self.add_template(dist_from_end="%s~%s" % span)

        assert len(self.templates) == len(set(self.templates))
    ############################

    ############################
    def extract_features(self, edus, edus_postag, sbnds, pbnds, span, n_edus):
        """
        :type edus: list of list of str
        :type edus_postag: list of list of str
        :type sbnds: list of (int, int)
        :type pbnds: list of (int, int)
        :type span: (int, int)
        :type n_edus: int
        :rtype: numpy.ndarray(shape=(1, feature_size), dtype=np.float32)
        """
        templates = self.generate_templates(edus=edus,
                                            edus_postag=edus_postag,
                                            sbnds=sbnds,
                                            pbnds=pbnds,
                                            span=span,
                                            n_edus=n_edus)
        template_dims = [self.template2dim.get(t, self.UNK_TEMPLATE_DIM) for t in templates] # list of int
        vector = utils.make_multihot_vectors(self.feature_size+1, [template_dims]) # (1, feature_size+1)
        vector = vector[:, :-1] # (1, feature_size
        return vector

    def extract_batch_features(self, edus, edus_postag, sbnds, pbnds, spans):
        """
        :type edus: list of list of str
        :type edus_postag: list of list of str
        :type sbnds: list of (int, int)
        :type pbnds: list of (int, int)
        :type spans: list of (int, int)
        :rtype: numpy.ndarray(shape=(N, feature_size), dtype=np.float32)
        """
        fire = [] # list of list of int
        n_edus = len(edus)
        for span in spans:
            templates = self.generate_templates(edus=edus,
                                                edus_postag=edus_postag,
                                                sbnds=sbnds,
                                                pbnds=pbnds,
                                                span=span,
                                                n_edus=n_edus)
            template_dims = [self.template2dim.get(t, self.UNK_TEMPLATE_DIM) for t in templates] # list of int
            fire.append(template_dims)
        vectors = utils.make_multihot_vectors(self.feature_size+1, fire) # (n_spans, feature_size+1)
        vectors = vectors[:, :-1] # (n_spans, feature_size)
        return vectors

    def generate_templates(self, edus, edus_postag, sbnds, pbnds, span, n_edus):
        """
        :type edus: list of list of str
        :type edus_postag: list of list of str
        :type sbnds: list of (int, int)
        :type pbnds: list of (int, int)
        :type span: (int, int)
        :type n_edus: int
        :rtype: list of str
        """
        templates = []

        b, e = span

        ###################
        # Features on N-grams
        # ngrams = self.extract_ngrams(edus[b], position="begin")
        # part_templates = [self.convert_to_template(lex_ngram_span_begin=ngram) for ngram in ngrams]
        # templates.extend(part_templates)
        #
        # ngrams = self.extract_ngrams(edus[e], position="end")
        # part_templates = [self.convert_to_template(lex_ngram_span_end=ngram) for ngram in ngrams]
        # templates.extend(part_templates)
        #
        # if b > 0:
        #     ngrams = self.extract_ngrams(edus[b-1], position="begin")
        #     part_templates = [self.convert_to_template(lex_ngram_left_context_begin=ngram) for ngram in ngrams]
        #     templates.extend(part_templates)
        #
        #     ngrams = self.extract_ngrams(edus[b-1], position="end")
        #     part_templates = [self.convert_to_template(lex_ngram_left_context_end=ngram) for ngram in ngrams]
        #     templates.extend(part_templates)
        #
        # if e < n_edus-1:
        #     ngrams = self.extract_ngrams(edus[e+1], position="begin")
        #     part_templates = [self.convert_to_template(lex_ngram_right_context_begin=ngram) for ngram in ngrams]
        #     templates.extend(part_templates)
        #
        #     ngrams = self.extract_ngrams(edus[e+1], position="end")
        #     part_templates = [self.convert_to_template(lex_ngram_right_context_end=ngram) for ngram in ngrams]
        #     templates.extend(part_templates)
        #
        # ngrams = self.extract_ngrams(edus_postag[b], position="begin")
        # part_templates = [self.convert_to_template(pos_ngram_span_begin=ngram) for ngram in ngrams]
        # templates.extend(part_templates)
        #
        # ngrams = self.extract_ngrams(edus_postag[e], position="end")
        # part_templates = [self.convert_to_template(pos_ngram_span_end=ngram) for ngram in ngrams]
        # templates.extend(part_templates)
        #
        # if b > 0:
        #     ngrams = self.extract_ngrams(edus_postag[b-1], position="begin")
        #     part_templates = [self.convert_to_template(pos_ngram_left_context_begin=ngram) for ngram in ngrams]
        #     templates.extend(part_templates)
        #
        #     ngrams = self.extract_ngrams(edus_postag[b-1], position="end")
        #     part_templates = [self.convert_to_template(pos_ngram_left_context_end=ngram) for ngram in ngrams]
        #     templates.extend(part_templates)
        #
        # if e < n_edus-1:
        #     ngrams = self.extract_ngrams(edus_postag[e+1], position="begin")
        #     part_templates = [self.convert_to_template(pos_ngram_right_context_begin=ngram) for ngram in ngrams]
        #     templates.extend(part_templates)
        #
        #     ngrams = self.extract_ngrams(edus_postag[e+1], position="end")
        #     part_templates = [self.convert_to_template(pos_ngram_right_context_end=ngram) for ngram in ngrams]
        #     templates.extend(part_templates)
        ###################

        # Features on size
        length_tokens_span = self.get_length_tokens_span(
                                    length=sum([len(edu) for edu in edus[b:e+1]]))
        template = self.convert_to_template(length_tokens="%s~%s" % length_tokens_span)
        templates.append(template)

        length_edus_span = self.get_length_edus_span(
                                    length=len(edus[b:e+1]))
        template = self.convert_to_template(length_edus="%s~%s" % length_edus_span)
        templates.append(template)

        length_sents_span = self.get_length_sents_span(
                                    length=sum([1 if b <= sb <= se <= e else 0 for sb, se in sbnds]))
        template = self.convert_to_template(length_sentences="%s~%s" % length_sents_span)
        templates.append(template)

        pbnds2 = [(sbnds[pb][0], sbnds[pe][1]) for pb, pe in pbnds]
        length_paras_span = self.get_length_paras_span(
                                    length=sum([1 if b <= pb <= pe <= e else 0 for pb, pe in pbnds2]))
        template = self.convert_to_template(length_paragraphs="%s~%s" % length_paras_span)
        templates.append(template)

        # Features on position
        distance_span = self.get_distance_span(b)
        template = self.convert_to_template(dist_from_begin="%s~%s" % distance_span)
        templates.append(template)

        distance_span = self.get_distance_span(n_edus - e - 1)
        template = self.convert_to_template(dist_from_end="%s~%s" % distance_span)
        templates.append(template)

        return templates

    def get_length_tokens_span(self, length):
        """
        :type length: int
        :rtype: (int, int)
        """
        for span_min, span_max in self.LENGTH_TOKENS_SPANS:
            if span_min <= length <= span_max:
                return (span_min, span_max)
        raise ValueError("Should never happan: length=%d" % length)

    def get_length_edus_span(self, length):
        """
        :type length: int
        :rtype: (int, int)
        """
        for span_min, span_max in self.LENGTH_EDUS_SPANS:
            if span_min <= length <= span_max:
                return (span_min, span_max)
        raise ValueError("Should never happan: length=%d" % length)

    def get_length_sents_span(self, length):
        """
        :type length: int
        :rtype: (int, int)
        """
        for span_min, span_max in self.LENGTH_SENTS_SPANS:
            if span_min <= length <= span_max:
                return (span_min, span_max)
        raise ValueError("Should never happan: length=%d" % length)

    def get_length_paras_span(self, length):
        """
        :type length: int
        :rtype: (int, int)
        """
        for span_min, span_max in self.LENGTH_PARAS_SPANS:
            if span_min <= length <= span_max:
                return (span_min, span_max)
        raise ValueError("Should never happan: length=%d" % length)

    def get_distance_span(self, distance):
        """
        :type distance: int
        :rtype: (int, int)
        """
        for span_min, span_max in self.DISTANCE_SPANS:
            if span_min <= distance <= span_max:
                return (span_min, span_max)
        raise ValueError("Should never happen: distance=%d" % distance)

