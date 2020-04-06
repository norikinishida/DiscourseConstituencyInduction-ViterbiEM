import pyprind

import treetk

def parse(model, decoder, databatch, path_pred):
    """
    :type model: Model
    :type decoder: IncrementalCKYDecoder
    :type databatch: DataBatch
    :type path_pred: str
    :rtype: None
    """
    with open(path_pred, "w") as f:
        prog_bar = pyprind.ProgBar(len(databatch))

        for edu_ids, edus, edus_postag, edus_head, sbnds, pbnds \
                in zip(databatch.batch_edu_ids,
                       databatch.batch_edus,
                       databatch.batch_edus_postag,
                       databatch.batch_edus_head,
                       databatch.batch_sbnds,
                       databatch.batch_pbnds):

            # Feature extraction
            edu_vectors = model.forward_edus(edus, edus_postag, edus_head) # (n_edus, bilstm_dim)
            padded_edu_vectors = model.pad_edu_vectors(edu_vectors) # (n_edus+2, bilstm_dim)
            mask_bwd, mask_fwd = model.make_masks() # (1, bilstm_dim), (1, bilstm_dim)

            # Parsing (constituency)
            unlabeled_sexp = decoder.decode(
                                    model=model,
                                    sexps=edu_ids,
                                    edus=edus,
                                    edus_postag=edus_postag,
                                    sbnds=sbnds,
                                    pbnds=pbnds,
                                    padded_edu_vectors=padded_edu_vectors,
                                    mask_bwd=mask_bwd,
                                    mask_fwd=mask_fwd,
                                    use_sbnds=True,
                                    use_pbnds=True) # list of str
            unlabeled_tree = treetk.sexp2tree(unlabeled_sexp, with_nonterminal_labels=False, with_terminal_labels=False)
            unlabeled_tree.calc_spans()
            unlabeled_spans = treetk.aggregate_spans(unlabeled_tree, include_terminal=False, order="pre-order") # list of (int, int)

            # Assigning majority labels to the unlabeled tree
            span2label = {(b,e): "<ELABORATION,N/S>" for (b,e) in unlabeled_spans}
            labeled_tree = treetk.assign_labels(unlabeled_tree, span2label, with_terminal_labels=False)
            labeled_sexp = treetk.tree2sexp(labeled_tree)

            f.write("%s\n" % " ".join(labeled_sexp))

            prog_bar.update()

