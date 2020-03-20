#!/usr/bin/env python3
"""
solve fracas problems

Hai Hu, Dec 2018
"""

# import infer, getMono
import sys, argparse, os, time
from infer import SentenceBase, Knowledge
from getMono import CCGtrees, CCGtree, ErrorCCGtree, ErrorCompareSemCat, eprint
from sklearn.metrics import confusion_matrix

def main():
    # -------------------------------------
    # parse cmd arguments
    description = """
    Solve FraCaS. Author: Hai Hu, huhai@indiana.edu
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-id', dest='fracas_id', type=str, default='all',
                        help='ID of the fracas problem to solve. E.g. "002", "035" '
                             "[default: %(default)s]")
    parser.add_argument('-sec', dest='sections', type=str, default='1',
                        help='sections of the fracas problem to solve. E.g. "1", "56" '
                             "[default: %(default)s]")
    parser.add_argument('-r', dest='n_rep', type=int, default=2,
                        help='number of replacements [default: %(default)s]')
    parser.add_argument('-k', dest='print_k', action='store_const', const=True, default=False,
                        help='if -k, print k')

    args = parser.parse_args()
    # -------------------------------------

    solveFracas(args.fracas_id, args.n_rep, args.print_k, args.sections)

def solveFracas(fracas_id, n_rep, print_k, sections):
    """  read in fracas  """
    start_time = time.time()

    if sections == "1":
        trees = CCGtrees("fracas_1_80.raw.tok.preprocess.log")
        trees.readEasyccgStr("fracas_1_80.raw.easyccg.parsed.txt")
        from fracas_index import IDX_P, IDX_H, UNDEFINED, ANSWERS

    elif sections == "56":
        trees = CCGtrees("fracas_sec_5_6.raw.tok.preprocess.log")
        trees.readEasyccgStr("fracas_sec_5_6.raw.easyccg.parsed.txt")
        from fracas_index import IDX_P_SEC_5_6 as IDX_P
        from fracas_index import IDX_H_SEC_5_6 as IDX_H
        from fracas_index import UNDEFINED_SEC_5_6 as UNDEFINED
        from fracas_index import ANSWERS_SEC_5_6 as ANSWERS

    else:
        print("wrong section! -sec can only be: 1, 56")
        exit()

    # --- sanity check ---
    # for p in IDX_P["055"]:
    #     print(trees.trees[p])
    # print(trees.trees[IDX_H["065"]])
    # --- sanity check: done ---

    # test one problem
    if fracas_id != 'all':
        solveFracas_one(fracas_id, trees, n_rep, print_k, IDX_P, IDX_H, UNDEFINED, ANSWERS)

    # all problems
    else:
        y_pred = []
        for fracas_id in sorted(IDX_P):  # solve
            if fracas_id in UNDEFINED: continue  # skipped undefined problems
            ans = solveFracas_one(fracas_id, trees, n_rep, print_k, IDX_P, IDX_H, UNDEFINED, ANSWERS)
            y_pred.append(ans)

        y_true = [ANSWERS[fracas_id] for fracas_id in sorted(ANSWERS) \
                  if fracas_id not in UNDEFINED]
        ids = [fracas_id for fracas_id in sorted(ANSWERS) \
               if fracas_id not in UNDEFINED]

        print('\ny_pred:', y_pred)
        print('y_true:', y_true)
        print(accuracy(ids, y_pred, y_true))

    print("\n\n--- %s seconds ---" % (time.time() - start_time))

def solveFracas_one(fracas_id, trees, n_rep, print_k, IDX_P, IDX_H, UNDEFINED, ANSWERS):
    """ sovle fracas problem #fracas_id
    steps:
        1. read in parsed Ps and H
        2. initialize knowledge base K, update K when reading in Ps and H
        3. do 3 replacement for each P, store all unique inferences in INF
        4. if H in INF: entail, else if: ... contradict, else: unknown
    """
    print('-'*30)
    print('\n*** solving fracas {} ***\n'.format(fracas_id))

    # build the trees here
    for p_idx in IDX_P[fracas_id]:
        trees.build_one_tree(p_idx, "easyccg")
    trees.build_one_tree(IDX_H[fracas_id], "easyccg")

    # -----------------------------
    # readin Ps and H
    Ps = [trees.trees[p_idx] for p_idx in IDX_P[fracas_id]]
    H = trees.trees[IDX_H[fracas_id]]

    # -----------------------------
    # initialize s
    s = SentenceBase()

    # -----------------------------
    # build knowledge
    k = Knowledge()
    k.build_quantifier(all_quant=True)         # all = every = each < some = a = an, etc.
    k.build_morph_tense()         # man = men, etc.

    # fix trees and update knowledge k, sentBase s
    # Ps
    for p in Ps:
        p.fixQuantifier()
        p.fixNot()
        # if parser == 'candc': t.fixRC()  # only fix RC for candc
        k.update_sent_pattern(p)  # patterns like: every X is NP
        k.update_word_lists(p)    # nouns, subSecAdj, etc.
        # if fracas_id in ["026", "027"]:
        #     k.update_quant_rel(p)
        s.add_P_str(p)

    # H
    H.fixQuantifier()
    H.fixNot()
    k.update_word_lists(H)      # need to find nouns, subSecAdjs, etc. in H
    s.add_H_str_there_be(H)     # transform ``there be'' in H

    k.update_modifier()         # adj + n < n, n + RC/PP < n, v + PP < v
    # k.update_rules()            # if N < VP and N < N_1, then N < N_1 who VP,
    # this gets us: 018,019,020

    s.k = k

    if print_k: k.print_knowledge()

    # -----------------------------
    # polarize
    print("\n*** polarizing ***\n")
    for p in s.Ps_ccgtree:  # Ps
        try:
            p.mark()
            p.polarize()
            p.getImpSign()
        except (ErrorCCGtree, ErrorCompareSemCat) as e: # , AssertionError
            print("cannot polarize:", p.wholeStr)
            print("error:", e)
            return type(e).__name__
        except AssertionError as e:
            print("assertion error!")
            print(e)
            p.printSent()
            exit()
        print("P: ", end="")
        p.printSent()
    print("H: ", end="")
    s.H_ccgtree.printSent()

    # -----------------------------
    # replacement
    # iterative deepening search (IDS)
    print("\n*** replacement ***\n")
    ans = s.solve_ids(depth_max=n_rep, fracas_id=fracas_id)

    print("\n--- tried ** {} ** inferences:".format(len(s.inferences)))
    for inf in sorted(s.inferences): print(inf)

    print("\n--- tried ** {} ** contradictions:".format(len(s.contras_str)))
    for contra in sorted(s.contras_str): print(contra)

    print("\n*** H: ", end="")
    print(s.H)

    # make decision
    print('\n*** decision ***')
    print('y_pred:', ans)
    print('y_true:', ANSWERS[fracas_id])
    return ans

def accuracy(ids, y_pred, y_true):
    """ compuate accuracy. pred and gold are lists """
    assert len(ids) == len(y_pred) == len(y_true)
    correct = 0
    incorrect = []
    print('\nno. pred true')
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]: correct += 1
        else: incorrect.append((ids[i], y_pred[i], y_true[i]))
        print(ids[i], y_pred[i], y_true[i])
    print('\nconfusion matrix')
    print(confusion_matrix(y_true, y_pred, labels=["E", "U", "C"]))
    print()
    print("incorrect predictions:")
    for i in incorrect: print(i)
    return round(correct / len(y_pred), 5)

def three_digit(num):
    if num < 10: return '00'+str(num)
    elif num < 100: return '0'+str(num)
    else: return str(num)

if __name__ == '__main__':
    main()





