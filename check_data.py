import treetk

import dataloader

def main():
    dataset = dataloader.read_rstdt(split="train", relation_level="coarse-grained", with_root=True)

    relation_mapper = treetk.rstdt.RelationMapper()
    i = 0
    for data in dataset:
        edu_ids = data.edu_ids
        edus = data.edus
        edus_postag = data.edus_postag
        edus_head = data.edus_head
        sbnds = data.sbnds
        pbnds = data.pbnds
        nary_sexp = data.nary_sexp
        bin_sexp = data.bin_sexp
        arcs = data.arcs

        print("Data instance #%d" % i)
        print("\t Paragraph #0")
        print("\t\t Sentence #0")
        print("\t\t\t EDU #0")
        print("\t\t\t\t EDU ID:", edu_ids[0])
        print("\t\t\t\t EDU:", edus[0])
        print("\t\t\t\t EDU (POS):", edus_postag[0])
        print("\t\t\t\t EDU (head):", edus_head[0])
        p_i = 1
        s_i = 1
        e_i = 1
        for p_begin, p_end in pbnds:
            print("\t Paragraph #%d" % p_i)
            for s_begin, s_end in sbnds[p_begin:p_end+1]:
                print("\t\t Sentence #%d" % s_i)
                for edu_id, edu, edu_postag, edu_head in zip(edu_ids[1+s_begin:1+s_end+1],
                                                             edus[1+s_begin:1+s_end+1],
                                                             edus_postag[1+s_begin:1+s_end+1],
                                                             edus_head[1+s_begin:1+s_end+1]):
                    print("\t\t\t EDU #%d" % e_i)
                    print("\t\t\t\t EDU ID:", edu_id)
                    print("\t\t\t\t EDU:", edu)
                    print("\t\t\t\t EDU (POS):", edu_postag)
                    print("\t\t\t\t EDU (head):", edu_head)
                    e_i += 1
                s_i += 1
            p_i += 1
        nary_tree = treetk.rstdt.postprocess(treetk.sexp2tree(nary_sexp, with_nonterminal_labels=True, with_terminal_labels=False))
        nary_tree = treetk.rstdt.map_relations(nary_tree, mode="c2a")
        bin_tree = treetk.rstdt.postprocess(treetk.sexp2tree(bin_sexp, with_nonterminal_labels=True, with_terminal_labels=False))
        bin_tree = treetk.rstdt.map_relations(bin_tree, mode="c2a")
        arcs = [(h,d,relation_mapper.c2a(l)) for h,d,l in arcs]
        dtree = treetk.arcs2dtree(arcs)
        treetk.pretty_print(nary_tree)
        treetk.pretty_print(bin_tree)
        treetk.pretty_print_dtree(dtree)
        i += 1

if __name__  =="__main__":
    main()
