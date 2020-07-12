import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

import utils

class SpanBasedModel(chainer.Chain):

    def __init__(self,
                 vocab_word,
                 vocab_postag,
                 vocab_deprel,
                 word_dim,
                 postag_dim,
                 deprel_dim,
                 lstm_dim,
                 mlp_dim,
                 initialW,
                 template_feature_extractor):
        """
        :type vocab_word: {str: int}
        :type vocab_postag: {str: int}
        :type vocab_deprel: {str: int}
        :type word_dim: int
        :type postag_dim: int
        :type deprel_dim: int
        :type lstm_dim: int
        :type mlp_dim: int
        :type initialW: numpy.ndarray(shape=(|V|, word_dim), dtype=np.float32)
        :type template_feature_extractor: TemplateFeatureExtractor
        """
        assert "<unk>" in vocab_word
        assert "<unk>" in vocab_deprel

        self.vocab_word = vocab_word
        self.vocab_postag = vocab_postag
        self.vocab_deprel = vocab_deprel

        # Word embedding
        self.word_dim = word_dim
        self.postag_dim = postag_dim
        self.deprel_dim = deprel_dim

        # BiLSTM over EDUs
        self.lstm_dim = lstm_dim
        self.bilstm_dim = lstm_dim + lstm_dim

        # Template features
        self.template_feature_extractor = template_feature_extractor
        self.tempfeat_dim = self.template_feature_extractor.feature_size

        # MLP
        self.mlp_dim = mlp_dim

        self.unk_word_id = self.vocab_word["<unk>"]
        self.unk_deprel_id = self.vocab_deprel["<unk>"]
        self.START_ID = np.asarray([0], dtype=np.int32) # (1,)
        self.STOP_ID = np.asarray([1], dtype=np.int32) # (1,)

        # Masks for BiLSTM's states
        self.mask_bwd = np.ones((1, self.bilstm_dim), dtype=np.float32) # (1, bilstm_dim)
        self.mask_fwd = np.ones((1, self.bilstm_dim), dtype=np.float32) # (1, bilstm_dim)
        self.mask_bwd[:, self.lstm_dim:] = 0.0 # Mask the latter part
        self.mask_fwd[:, :self.lstm_dim] = 0.0 # Mask the former part

        links = {}
        # EDU embedding
        links["embed_word"] = L.EmbedID(len(self.vocab_word),
                                        self.word_dim,
                                        ignore_label=-1,
                                        initialW=initialW)
        links["embed_postag"] = L.EmbedID(len(self.vocab_postag),
                                        self.postag_dim,
                                        ignore_label=-1,
                                        initialW=None)
        links["embed_deprel"] = L.EmbedID(len(self.vocab_deprel),
                                        self.deprel_dim,
                                        ignore_label=-1,
                                        initialW=None)
        links["W_edu"] = L.Linear(self.word_dim + self.postag_dim +
                                  self.word_dim + self.postag_dim +
                                  self.word_dim + self.postag_dim + self.deprel_dim,
                                  self.word_dim)
        # BiLSTM
        links["bilstm"] = L.NStepBiLSTM(n_layers=1,
                                        in_size=self.word_dim,
                                        out_size=self.lstm_dim,
                                        dropout=0.0)
        # Boundary embedding
        links["embed_boundary"] = L.EmbedID(2, self.bilstm_dim)
        # MLPs
        links["W1_bkt"] = L.Linear(self.bilstm_dim + self.tempfeat_dim, self.mlp_dim)
        links["W2_bkt"] = L.Linear(self.mlp_dim, 1)
        super(SpanBasedModel, self).__init__(**links)

    def forward_edus(self, edus, edus_postag, edus_head):
        """
        :type edus: lsit of list of str
        :type edus_postag: list of list of str
        :type edus_head: list of (str, str, str)
        :rtype: Variable(shape=(n_edus, bilstm_dim), dtype=np.float32)
        """
        with chainer.using_config("train", False), chainer.no_backprop_mode():
            #################
            # TODO?
            # Bag-of-word embedding
            # word_ids = [[self.vocab_word.get(w, self.unk_word_id) for w in edu]
            #             for edu in edus] # n_edus * length * int
            # word_ids, mask = utils.padding(word_ids, head=True, with_mask=True) # (n_edus, max_length), (n_edus, max_length)
            # n_edus, max_length = word_ids.shape
            # word_ids = utils.convert_ndarray_to_variable(word_ids, seq=False) # (n_edus, max_length)
            # mask = utils.convert_ndarray_to_variable(mask, seq=False) # (n_edus, max_length)
            # word_ids = F.reshape(word_ids, (n_edus * max_length,)) # (n_edus * max_length,)
            # word_vectors = F.dropout(self.embed_word(word_ids), ratio=0.2) # (n_edus * max_length, word_dim)
            # word_vectors = F.reshape(word_vectors, (n_edus, max_length, self.word_dim)) # (n_edus, max_length, word_dim)
            # mask = F.broadcast_to(mask[:,:,None], (n_edus, max_length, self.word_dim)) # (n_edus, max_length, word_dikm)
            # word_vectors = word_vectors * mask # (n_edus, max_length, word_dim)
            # bow_vectors = F.sum(word_vectors, axis=1) # (n_edus, word_dim)
            #################

            # Beginning-word embedding
            begin_word_ids = [self.vocab_word.get(edu[0], self.unk_word_id) for edu in edus] # n_edus * int
            begin_word_ids = np.asarray(begin_word_ids, dtype=np.int32) # (n_edus,)
            begin_word_ids = utils.convert_ndarray_to_variable(begin_word_ids, seq=False) # (n_edus,)
            begin_word_vectors = F.dropout(self.embed_word(begin_word_ids), ratio=0.2) # (n_edus, word_dim)

            # End-word embedding
            end_word_ids = [self.vocab_word.get(edu[-1], self.unk_word_id) for edu in edus] # n_edus * int
            end_word_ids = np.asarray(end_word_ids, dtype=np.int32) # (n_edus,)
            end_word_ids = utils.convert_ndarray_to_variable(end_word_ids, seq=False) # (n_edus,)
            end_word_vectors = F.dropout(self.embed_word(end_word_ids), ratio=0.2) # (n_edus, word_dim)

            # Head-word embedding
            head_word_ids = [self.vocab_word.get(head_word, self.unk_word_id)
                            for (head_word, head_postag, head_deprel) in edus_head] # n_edus * int
            head_word_ids = np.asarray(head_word_ids, dtype=np.int32) # (n_edus,)
            head_word_ids = utils.convert_ndarray_to_variable(head_word_ids, seq=False) # (n_edus,)
            head_word_vectors = F.dropout(self.embed_word(head_word_ids), ratio=0.2) # (n_edus, word_dim)

        # Beginning-postag embedding
        begin_postag_ids = [self.vocab_postag[edu_postag[0]] for edu_postag in edus_postag] # n_edus * int
        begin_postag_ids = np.asarray(begin_postag_ids, dtype=np.int32) # (n_edus,)
        begin_postag_ids = utils.convert_ndarray_to_variable(begin_postag_ids, seq=False) # (n_edus,)
        begin_postag_vectors = F.dropout(self.embed_postag(begin_postag_ids), ratio=0.2) # (n_edus, postag_dim)

        # End-postag embedding
        end_postag_ids = [self.vocab_postag[edu_postag[-1]] for edu_postag in edus_postag] # n_edus * int
        end_postag_ids = np.asarray(end_postag_ids, dtype=np.int32) # (n_edus,)
        end_postag_ids = utils.convert_ndarray_to_variable(end_postag_ids, seq=False) # (n_edus,)
        end_postag_vectors = F.dropout(self.embed_postag(end_postag_ids), ratio=0.2) # (n_edus, postag_dim)

        # Head-postag embedding
        head_postag_ids = [self.vocab_postag[head_postag]
                      for (head_word, head_postag, head_deprel) in edus_head] # n_edus * int
        head_postag_ids = np.asarray(head_postag_ids, dtype=np.int32) # (n_edus,)
        head_postag_ids = utils.convert_ndarray_to_variable(head_postag_ids, seq=False) # (n_edus,)
        head_postag_vectors = F.dropout(self.embed_postag(head_postag_ids), ratio=0.2) # (n_edus, postag_dim)

        # Head-deprel embedding
        head_deprel_ids = [self.vocab_deprel.get(head_deprel, self.unk_deprel_id)
                      for (head_word, head_postag, head_deprel) in edus_head] # n_edus * int
        head_deprel_ids = np.asarray(head_deprel_ids, dtype=np.int32) # (n_edus,)
        head_deprel_ids = utils.convert_ndarray_to_variable(head_deprel_ids, seq=False) # (n_edus,)
        head_deprel_vectors = F.dropout(self.embed_deprel(head_deprel_ids), ratio=0.2) # (n_edus, deprel_dim)

        # Concat
        edu_vectors = F.concat([begin_word_vectors,
                                end_word_vectors,
                                head_word_vectors,
                                begin_postag_vectors,
                                end_postag_vectors,
                                head_postag_vectors,
                                head_deprel_vectors],
                                axis=1) # (n_edus, 3 * word_dim + 3 * postag_dim + deprel_dim)
        edu_vectors = F.relu(self.W_edu(edu_vectors)) # (n_edus, word_dim)

        # BiLSTM
        h_init, c_init = None, None
        _, _, states = self.bilstm(hx=h_init, cx=c_init, xs=[edu_vectors]) # (1, n_edus, bilstm_dim)
        edu_vectors = states[0] # (n_edus, bilstm_dim)

        return edu_vectors

    def pad_edu_vectors(self, edu_vectors):
        """
        :type edu_vectors: Variable(shape=(n_edus, bilstm_dim), dtype=np.float32)
        :rtype: Variable(shape=(n_edus+2, bilstm_dim), dtype=np.float32)
        """
        start_id = utils.convert_ndarray_to_variable(self.START_ID, seq=False) # (1,)
        stop_id = utils.convert_ndarray_to_variable(self.STOP_ID, seq=False) # (1,)
        start_vector = self.embed_boundary(start_id) # (1, bilstm_dim)
        stop_vector = self.embed_boundary(stop_id) # (1, bilstm_dim)
        padded_edu_vectors = F.vstack([start_vector, edu_vectors, stop_vector]) # (n_edus+2, bilstm_dim)
        return padded_edu_vectors

    def make_masks(self):
        """
        :rtype: Variable(shape=(1, bilstm_dim), dtype=np.float32), Variable(shape=(1, bilstm_dim), dtype=np.float32)
        """
        mask_bwd = utils.convert_ndarray_to_variable(self.mask_bwd, seq=False) # (1, bilstm_dim)
        mask_fwd = utils.convert_ndarray_to_variable(self.mask_fwd, seq=False) # (1, bilstm_dim)
        return mask_bwd, mask_fwd

    def compute_span_vectors(
                self,
                edus,
                edus_postag,
                sbnds,
                pbnds,
                padded_edu_vectors,
                mask_bwd,
                mask_fwd,
                batch_spans):
        """
        :type edus: list of list of str
        :type edus_postag: list of list of str
        :type sbnds: list of (int, int)
        :type pbnds: list of (int, int)
        :type padded_edu_vectors: Variable(shape=(n_edus+2, bilstm_dim), dtype=np.float32)
        :type mask_bwd: Variable(shape=(1, bilstm_dim), dtype=np.float32)
        :type mask_fwd: Variable(shape=(1, bilstm_dim), dtype=np.float32)
        :type batch_spans: list of list of (int, int)
        :rtype: Variable(shape=(batch_size * n_spans, bilstm_dim + tempfeat_dim), dtype=np.float32)
        """
        batch_size = len(batch_spans)
        n_spans = len(batch_spans[0])
        total_spans = batch_size * n_spans
        for spans in batch_spans:
            assert len(spans) == n_spans

        # Reshape
        flatten_batch_spans = utils.flatten_lists(batch_spans) # total_spans * (int, int)
        # NOTE that indices in batch_spans should be shifted by +1 due to the boundary padding
        bm1_indices = [(b-1)+1 for b,e in flatten_batch_spans] # total_spans * int
        b_indices = [b+1 for b,e in flatten_batch_spans] # total_spans * int
        e_indices = [e+1 for b,e in flatten_batch_spans] # total_spans * int
        ep1_indices = [(e+1)+1 for b,e in flatten_batch_spans] # total_spans * int

        # Feature extraction
        bm1_padded_edu_vectors = F.get_item(padded_edu_vectors, bm1_indices) # (total_spans, bilstm_dim)
        b_padded_edu_vectors = F.get_item(padded_edu_vectors, b_indices) # (total_spans, bilstm_dim)
        e_padded_edu_vectors = F.get_item(padded_edu_vectors, e_indices) # (total_spans, bilstm_dim)
        ep1_padded_edu_vectors = F.get_item(padded_edu_vectors, ep1_indices) # (total_spans, bilstm_dim)
        mask_bwd = F.broadcast_to(mask_bwd, (total_spans, self.bilstm_dim)) # (total_spans, bilstm_dim)
        mask_fwd = F.broadcast_to(mask_fwd, (total_spans, self.bilstm_dim)) # (total_spans, bilstm_dim)
        span_vectors = mask_bwd * (e_padded_edu_vectors - bm1_padded_edu_vectors) \
                        + mask_fwd * (b_padded_edu_vectors - ep1_padded_edu_vectors) # (total_spans, bilstm_dim)

        # Template features
        tempfeat_vectors = self.template_feature_extractor.extract_batch_features(
                                        edus=edus,
                                        edus_postag=edus_postag,
                                        sbnds=sbnds,
                                        pbnds=pbnds,
                                        spans=flatten_batch_spans) # (total_spans, tempfeat_dim)
        tempfeat_vectors = utils.convert_ndarray_to_variable(tempfeat_vectors, seq=False) # (total_spans, tempfeat_dim)
        span_vectors = F.concat([span_vectors, tempfeat_vectors], axis=1) # (total_spans, bilstm_dim + tempfeat_dim)

        return span_vectors

    def forward_spans_for_bracketing(
                self,
                edus,
                edus_postag,
                sbnds,
                pbnds,
                padded_edu_vectors,
                mask_bwd,
                mask_fwd,
                batch_spans,
                aggregate):
        """
        :type edus: list of list of str
        :type edus_postag: list of list of str
        :type sbnds: list of (int, int)
        :type pbnds: list of (int, int)
        :type padded_edu_vectors: Variable(shape=(n_edus+2, bilstm_dim), dtype=np.float32)
        :type mask_bwd: Variable(shape=(1, bilstm_dim), dtype=np.float32)
        :type mask_fwd: Variable(shape=(1, bilstm_dim), dtype=np.float32)
        :type batch_spans: list of list of (int, int)
        :type aggregate: bool
        :rtype: Variable(shape=(batch_size, n_spans, 1)/(batch_size, 1), dtype=np.float32)
        """
        batch_size = len(batch_spans)
        n_spans = len(batch_spans[0])
        # total_spans = batch_size * n_spans
        for spans in batch_spans:
            assert len(spans) == n_spans

        # Feature extraction
        span_vectors = self.compute_span_vectors(
                                            edus=edus,
                                            edus_postag=edus_postag,
                                            sbnds=sbnds,
                                            pbnds=pbnds,
                                            padded_edu_vectors=padded_edu_vectors,
                                            mask_bwd=mask_bwd,
                                            mask_fwd=mask_fwd,
                                            batch_spans=batch_spans) # (total_spans, bilstm_dim + tempfeat_dim)

        # MLP (Constituency Scoring)
        span_scores = self.W2_bkt(F.dropout(F.relu(self.W1_bkt(span_vectors)), ratio=0.4)) # (total_spans, 1)
        span_scores = F.reshape(span_scores, (batch_size, n_spans, 1)) # (batch_size, n_spans, 1)

        if aggregate:
            tree_scores = F.sum(span_scores, axis=1) # (batch_size, 1)
            return tree_scores
        else:
            return span_scores

