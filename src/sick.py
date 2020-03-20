#!/usr/bin/env python3
"""
solve sick problems

Hai Hu, Jan 2019
"""

# import infer, getMono
import sys, argparse, os, time, csv
from infer import SentenceBase, Knowledge
from getMono import CCGtrees, CCGtree, ErrorCCGtree, ErrorCompareSemCat, eprint
from sklearn.metrics import confusion_matrix
import sick_ans
import spacy
from pass2act import P2A_transformer
import mywordnet

# setup logger: https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings
import logging
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
def setup_logger(name, log_file, level=logging.INFO, formatter=formatter):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file+"_"+time.strftime('%Y_%m_%d-%H_%M_%S'), mode="w+")        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def main():
    # -------------------------------------
    # parse cmd arguments
    description = """
    Solve sick. Author: Hai Hu, huhai@indiana.edu
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-id', dest='sick_id', type=str, nargs='+', default=['all'],
                        help='ID of the sick problem to solve. E.g. 1, 23. If "trial", then '
                             'try all cases of trial that are E or C '
                             "[default: %(default)s]")
    parser.add_argument('-r', dest='n_rep', type=int, default=2,
                        help='number of replacements [default: %(default)s]')
    parser.add_argument('-k', dest='print_k', action='store_const', const=True, default=False,
                        help='if -k, print k [default: %(default)s]')
    parser.add_argument('-g', dest='gen_inf', action='store_const', const=True, default=False,
                        help='if -g, generate infs, neutrals and contras, do not solve the problems [default: %(default)s]')
    parser.add_argument('-s', dest='start', type=int, default=None,
                        help='start of id to test [default: %(default)s]')
    parser.add_argument('-e', dest='end', type=int, default=None,
                        help='end of id to test [default: %(default)s]')
    parser.add_argument('-l', dest='pred_log', action='store_const', const=True, default=False,
                        help='if -l, then save predictions in a log file: pred_monalog_DATE_TIME [default: %(default)s]')
    parser.add_argument('-b', dest='backward', action='store_const', const=True, default=False,
                        help='if -b, then reverse P and H = backward [default: %(default)s]')

    args = parser.parse_args()
    # -------------------------------------

    # sick2uniq = sick_id_2_uniq_id()
    sick2uniq = sick_ans.sick2uniq
    # print(P_idx(50, sick2uniq))
    # print(H_idx(50, sick2uniq))

    if args.start and args.end:
        args.sick_id = list(range(args.start, args.end+1))

    solver = SICK_solver(args.sick_id, args.n_rep, args.print_k, sick2uniq, args.gen_inf, args.pred_log, args.backward)

    solver.solveSick()

class SICK_solver():
    def __init__(self, sick_id, n_rep, print_k, sick2uniq, gen_inf, pred_log, backward):
        self.sick_id = sick_id
        self.n_rep = n_rep
        self.print_k = print_k
        self.sick2uniq = sick2uniq
        self.ANSWERS = sick_ans.ANSWERS_new  # or sick_ans.ANSWERS
        # fix ANSWERS
        for idx in sick_ans.ANSWERS_HU:
            self.ANSWERS[idx] = sick_ans.ANSWERS_HU[idx]
        self.p2a = P2A_transformer(spacy.load('en'))
        self.gen_inf = gen_inf
        self.nlp = spacy.load('en_core_web_lg')  # make sure to use larger model!
        self.pred_log = pred_log
        if self.pred_log: self.pred_logger = setup_logger('pred_logger', 'pred_monalog.txt', formatter="")
        self.reverse_P_H = backward  # do backward inference

    def P_idx(self, sick_id):
        """ return idx of P in sick2uniq """
        if self.reverse_P_H: return self.sick2uniq[sick_id]["H"]
        return self.sick2uniq[sick_id]["P"]

    def H_idx(self, sick_id):
        """ return idx of H in sick2uniq """
        if self.reverse_P_H: return self.sick2uniq[sick_id]["P"]
        return self.sick2uniq[sick_id]["H"]

    def solveSick(self):
        start_time = time.time()

        trees = CCGtrees(fn_log="sick_uniq.raw.tok.preprocess.log")

        # read parsed trees from different parsers
        # trees.readEasyccgStr("sick_uniq.raw.easyccg.parsed.txt")
        trees.readEasyccgStr("sick_uniq.raw.depccg.parsed.txt")
        
        # --- sanity check ---
        # for p in IDX_P["055"]:
        #     print(trees.trees[p])
        # print(trees.trees[IDX_H["065"]])
        # --- sanity check: done ---

        # trials
        if self.sick_id[0] == 'trial':
            sick_ids = sick_ans.ids_trial_E_C    # ids_trial_E_C
            sick_ids = sick_ans.ids_trial
        elif self.sick_id[0] == 'wrongs': sick_ids = sick_ans.ids_wrongs_U_test
        elif self.sick_id[0] == 'trial_c': sick_ids = sick_ans.ids_trial_C
        elif self.sick_id[0] == 'trial_e': sick_ids = sick_ans.ids_trial_E

        # test on train data
        elif self.sick_id[0] == 'train': sick_ids = sick_ans.ids_train[:1000]

        # test on test
        elif self.sick_id[0] == 'test': sick_ids = sick_ans.ids_test[:]

        # test on one word diff
        elif self.sick_id[0] == 'onediff':
            ids_one_diff = set(sick_ans.ids_one_diff)
            sick_ids = [i for i in sick_ans.ids_train if i in ids_one_diff]

        # test one or more problems
        elif self.sick_id[0] != 'all': sick_ids = [int(i) for i in self.sick_id if int(i) in self.sick2uniq]

        # all problems
        else: sick_ids = sorted(self.sick2uniq)

        self.solveSick_helper(sick_ids, trees)

        print("\n\n--- %s seconds ---" % (time.time() - start_time))

    def solveSick_helper(self, sick_ids, trees):
        """ sick_id: a list of ids to solve """
        y_pred = []

        fout = None
        save_sents = False
        # save the polarized trees to later inspection
        if save_sents: fout = open("sick_trees_polarized.csv", 'w')

        for id_to_solve in sick_ids:
            # only print P and H
            # ans = self.solveSick_one(id_to_solve, trees, reverse=False, fout=fout)
            # continue

            # assume all the "U" can be correctly solved
            # if self.ANSWERS[id_to_solve] == "U":
            #     y_pred.append("U")
            #     continue
            ans = self.map_ans(self.solveSick_one(id_to_solve, trees, reverse=False, fout=fout))
            # if the ans is "U", try going from hypothesis to premise, see if get "C"
            if not self.gen_inf:  # if not only generate infs and contras
                if ans == "U":
                    ans_rev = self.map_ans(self.solveSick_one(id_to_solve, trees, reverse=True, fout=fout))
                    if ans_rev in ["C", "E_pass"]:  # reverse will only count when E_pass
                        ans = ans_rev
                if ans == "E_pass": ans = "E"  # E_pass = passive to active
            y_pred.append(ans)
            if self.pred_log: self.pred_logger.info("{}\t{}".format(id_to_solve, ans))

        y_true = [self.ANSWERS[i] for i in sick_ids]
        print('\ny_pred:', y_pred)
        print('y_true:', y_true)
        print('acc:')
        print(self.accuracy(sick_ids, y_pred, y_true))
        if save_sents: fout.close()

    def solveSick_one(self, id_to_solve, trees, reverse=False, fout=None):
        """ solve sick problem #sick_id
        steps:
            1. read in parsed Ps and H
            2. initialize knowledge base K, update K when reading in Ps and H
            3. do 3 replacement for each P, store all unique inferences in INF
            4. if H in INF: entail, else if: ... contradict, else: unknown
        """
        print('-'*30)
        print('\n*** solving sick {} ***\n'.format(id_to_solve))

        # -----------------------------
        # readin Ps and H
        # build the tree here
        use_lemma = True
        if self.gen_inf: use_lemma = False
        if not reverse:
            P = trees.build_one_tree(self.P_idx(id_to_solve), "easyccg", use_lemma)
            H = trees.build_one_tree(self.H_idx(id_to_solve), "easyccg", use_lemma)
        else:
            P = trees.build_one_tree(self.H_idx(id_to_solve), "easyccg", use_lemma)
            H = trees.build_one_tree(self.P_idx(id_to_solve), "easyccg", use_lemma)

        # just print P, H
        # print("{}\t{}\t{}".format(id_to_solve, P.printSent_raw_no_pol(stream=sys.stderr).strip(), H.printSent_raw_no_pol(stream=sys.stderr)))
        # return

        # -----------------------------
        # passive to active
        # my_act = self.p2a.pass2act(P.printSent_raw_no_pol(stream=sys.stdout, verbose=False))
        # my_act = my_act.rstrip('. ').upper()
        # print('active:')
        # print(my_act)
        # if my_act == H.printSent_raw_no_pol(stream=sys.stdout, verbose=False).strip().upper():
        #     ans = "E_pass"
        #     print('active = passive')
        #     print('\n*** decision ***')
        #     print('y_pred:', ans)
        #     print('y_true:', self.ANSWERS[id_to_solve])
        #     return ans

        # -----------------------------
        # initialize s
        s = SentenceBase(gen_inf=self.gen_inf)

        # -----------------------------
        # build knowledge
        k = Knowledge()
        k.build_quantifier()         # all = every = each < some = a = an, etc.
        # k.build_morph_tense()         # man = men, etc.

        # fix trees and update knowledge k, sentBase s
        # P
        P.fixQuantifier()
        P.fixNot()
        # if parser == 'candc': t.fixRC()  # only fix RC for candc
        k.update_sent_pattern(P)  # patterns like: every X is NP
        k.update_word_lists(P)    # nouns, subSecAdj, etc.
        s.add_P_str(P)

        # H
        H.fixQuantifier()
        H.fixNot()
        k.update_word_lists(H)      # need to find nouns, subSecAdjs, etc. in H
        s.add_H_str_there_be(H)     # transform ``there be'' in H

        # -----------------------------
        k.build_manual_for_sick()
        mywordnet.assign_all_relations_wordnet(k)
        if self.gen_inf:  # todo: not implemented??
            mywordnet.add_alternations(sent_id=id_to_solve, nlp=self.nlp, k=k)
        # -----------------------------

        k.update_modifier()         # adj + n < n, n + RC/PP < n, v + PP < v
        # k.update_rules()            # if N < VP and N < N_1, then N < N_1 who VP,

        s.k = k

        if self.print_k: k.print_knowledge()

        # exit()
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
                return type(e).__name__
                # exit()
            # print("P: ", end="")
            # p.printSent_raw_no_pol()
            print("P: ", end="")
            p.printSent()
        print("H: ", end="")
        s.H_ccgtree.printSent_raw_no_pol()
        # s.H_ccgtree.printSent()

        # -----------------------------
        # replacement
        # iterative deepening search (IDS)
        print("\n*** replacement ***\n")
        ans = s.solve_ids(depth_max=self.n_rep, fracas_id=None)

        print("\n--- tried ** {} ** inferences:".format(len(s.inferences)))
        # for inf in sorted(s.inferences): print(inf)
        for inf in s.inferences_tree: inf.printSent_raw()

        print("\n--- tried ** {} ** contradictions:".format(len(s.contras_str)))
        for contra in sorted(s.contras_str): print(contra.lower())

        print("\n*** H: ", end="")
        print(s.H)

        # make decision
        print('\n*** decision ***')
        print('y_pred:', ans)
        print('y_true:', self.ANSWERS[id_to_solve])

        printOut = False
        truth = self.ANSWERS[id_to_solve]
        if not reverse and truth == "E" and ans != "E":
            printOut = True
        if truth == "C" and ans != "C":
            printOut = True

        if fout and printOut:
            myformat_rev = "{}_r\t{}\t{}\t{}\t{}\n"
            myformat = "{}\t{}\t{}\t{}\t{}\n"
            if reverse:
                fout.write(myformat_rev.format(id_to_solve, P.printSent_raw(),
                                                  H.printSent_raw_no_pol(), self.ANSWERS[id_to_solve], ans))
            else: fout.write(myformat.format(id_to_solve, P.printSent_raw(),
                                                  H.printSent_raw_no_pol(), self.ANSWERS[id_to_solve], ans))

        return ans

    def accuracy(self, ids, y_pred, y_true):
        """ compuate accuracy. pred and gold are lists """
        assert len(ids) == len(y_pred) == len(y_true)
        correct = 0
        incorrect = []
        print('\npred true no.')
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]: correct += 1
            else: incorrect.append((ids[i], y_pred[i], y_true[i]))
            print(y_pred[i], y_true[i], ids[i])
        print('\nconfusion matrix')
        print(confusion_matrix(y_true, y_pred, labels=["E", "U", "C"]))
        print()
        print("incorrect predictions:\nid, pred, truth")
        for i in incorrect: print(i)
        return round(correct / len(y_pred), 5)

    def map_ans(self, ans):
        """ change errors to Neutral """
        # if ans.startswith("Error"): return "U"
        return ans

if __name__ == '__main__':
    main()
