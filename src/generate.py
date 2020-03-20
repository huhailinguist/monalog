#!/usr/bin/env python3
"""
generate inferences, contradictions and neutrals for SICK

Hai Hu
"""

from getMono import CCGtrees, CCGtree, ErrorCCGtree, ErrorCompareSemCat, eprint
from infer import Knowledge, SentenceBase
import sick_ans, spacy, sys, time
# from pass2act import P2A_transformer
from utils import P_idx, H_idx


def main():
    USAGE="""
    generate inferences for SICK
    usage:
        python generate.py chunk_id
        chunk_id={trial, 1-9}
    """
    if len(sys.argv) != 2:
        print(USAGE)
        exit()
    elif sys.argv[1] not in ['trial'] + list(range(1,10)):
        print(USAGE)
        exit()      
    start_time = time.time()
    generate(sys.argv[1])
    eprint("\n\n--- %s seconds ---" % (time.time() - start_time))

def generate(chunk):
    n_rep = 3

    trees = CCGtrees("sick_uniq.raw.tok.preprocess.log")
    trees.readEasyccgStr("sick_uniq.raw.easyccg.parsed.txt")

    if chunk == "trial": sick_ids = sick_ans.ids_trial_E_C
    # sick_ids = [1237]
    if chunk == "1": sick_ids = sick_ans.ids_train[:500]
    elif chunk == "2": sick_ids = sick_ans.ids_train[500:1000]
    elif chunk == "3": sick_ids = sick_ans.ids_train[1000:1500]
    elif chunk == "4": sick_ids = sick_ans.ids_train[1500:2000]
    elif chunk == "5": sick_ids = sick_ans.ids_train[2000:2500]
    elif chunk == "6": sick_ids = sick_ans.ids_train[2500:3000]
    elif chunk == "7": sick_ids = sick_ans.ids_train[3000:3500]
    elif chunk == "8": sick_ids = sick_ans.ids_train[3500:4000]
    elif chunk == "9": sick_ids = sick_ans.ids_train[4000:4500]
    # elif chunk == 10: sick_ids = sick_ans.ids_train[4500:]

    for id_to_solve in sick_ids:
        P = trees.build_one_tree(P_idx(sick_ans.sick2uniq, id_to_solve),
                                 "easyccg", use_lemma=True)
        H = trees.build_one_tree(H_idx(sick_ans.sick2uniq, id_to_solve),
                                 "easyccg", use_lemma=True)

        # -----------------------------
        # passive to active
        # my_act = p2a.pass2act(P.printSent_raw_no_pol(stream=sys.stdout, verbose=False))
        # my_act = my_act.rstrip('. ')
        # eprint('original:', P.printSent_raw_no_pol(stream=sys.stdout, verbose=False))
        # eprint('active:', my_act)

        # -----------------------------
        # initialize s
        s = SentenceBase(gen_inf=True)

        # -----------------------------
        # build knowledge
        k = Knowledge()
        k.build_manual_for_sick()  # TODO
        k.build_quantifier(all_quant=False)         # all = every = each < some = a = an, etc.
        k.build_morph_tense()         # man = men, etc.

        # fix trees and update knowledge k, sentBase s
        # P
        P.fixQuantifier()
        P.fixNot()
        k.update_sent_pattern(P)  # patterns like: every X is NP
        k.update_word_lists(P)    # nouns, subSecAdj, etc.
        s.add_P_str(P)

        # H
        H.fixQuantifier()
        H.fixNot()
        k.update_word_lists(H)      # need to find nouns, subSecAdjs, etc. in H
        s.add_H_str_there_be(H)     # transform ``there be'' in H

        k.update_modifier()         # adj + n < n, n + RC/PP < n, v + PP < v

        s.k = k

        # -----------------------------
        # polarize
        eprint("\n*** polarizing ***\n")
        for p in s.Ps_ccgtree:  # Ps
            try:
                p.mark()
                p.polarize()
                p.getImpSign()
            except (ErrorCCGtree, ErrorCompareSemCat) as e:  # , AssertionError
                eprint("cannot polarize:", p.wholeStr)
                eprint("error:", e)
            except AssertionError as e:
                eprint("assertion error!")
                eprint(e)
                p.printSent()
                exit()
            eprint("P: ", end="")
            p.printSent_raw_no_pol(stream=sys.stderr)
        eprint("H: ", end="")
        s.H_ccgtree.printSent_raw_no_pol(stream=sys.stderr)

        eprint("\n*** replacement ***\n")
        try:
            ans = s.solve_ids(depth_max=n_rep, fracas_id=None)
        except (ErrorCompareSemCat, ErrorCCGtree):
            continue

        eprint("\n--- tried ** {} ** inferences:".format(len(s.inferences)))
        # for inf in sorted(s.inferences): print(inf)
        for inf in s.inferences_tree:
            print("{}\t{}\t{}\t{}".format(
                id_to_solve,
                P.printSent_no_pol(stream=sys.stdout),
                inf.printSent_no_pol(stream=sys.stdout),
                "ENTAILMENT"
            ))

        eprint("\n--- tried ** {} ** contradictions:".format(len(s.contras_str)))
        for contra in sorted(s.contras_str):
            print("{}\t{}\t{}\t{}".format(
                id_to_solve,
                P.printSent_no_pol(stream=sys.stdout),
                contra.lower(),
                "CONTRADICTION"
            ))

        # TODO neutral
        # eprint("\n--- tried ** {} ** neutrals:".format(len(s.neutrals_tree)))
        # for neutral in s.neutrals_tree:
        #     print("{}\t{}\t{}\t{}".format(
        #         id_to_solve,
        #         P.printSent_no_pol(stream=sys.stdout),
        #         neutral.printSent_no_pol(stream=sys.stdout),
        #         "NEUTRAL"
        #     ))

        # TODO if P and H contradicts, then generate infs from H, they contradict P


if __name__ == '__main__':
    main()










