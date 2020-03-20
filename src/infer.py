#!/usr/bin/env python3

# get inference
# input: polarized trees from getMono.py
# Hai Hu, Feb 2018


import getMono, utils
import sys, os, argparse
from copy import deepcopy
from logical_system import LogicalSystem, LogicalSentence
from getMono import eprint
from utils import node_at_least_3, node_at_least_several
from utils import find_head_noun, same_ani_noun, make_nontermnode,\
    set_parent_children
### importsbelow only needed for generate_challenge.py, which
### is only needed for generating data for semantics fragment paper at AAAI 2020
# from generate_challenge import NP_ani, node_and_for_NP

# non-subsective adjectives
NON_SUBSEC_ADJ = ["fake", "so-called", "pseudo", "false"]

THERE_BE = ["THERE ARE", "THERE IS", "THERE WAS", "THERE WERE", "THERE BE"]

QUANTIFIERS = ["AN", "A", "SOME", "ALL", "EVERY", "EACH", "FEW", "MOST", "NO"]


def main():
    # parse cmd arguments
    description = """
    Inference from premises by replacement. Author: Hai Hu, huhai@indiana.edu
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', '--sentNo', dest='sentNo', type=str, nargs='+', default=['all'],
                        help='index(s) of sentence to process. E.g. "2", "3 5", "all" ' \
                             "[default: %(default)s]")
    # parser.add_argument('-p', dest='parser', default='candc', choices=['candc', 'easyccg'],
    #                     help='parser of your choice: candc, easyccg ' \
    #                          "[default: %(default)s]")
    parser.add_argument('-v', dest='verbose', choices=[-1,0,1,2,3], type=int, default=-1,
                        help='verbose: -1: None, 0: after fixTree(), '
                             '1: after mark(), 2: after polarize(), \n3: all ' \
                             "[default: %(default)s]")
    parser.add_argument('-t', dest='test', action='store_const', const=True, default=False,
                        help='if -t, run test()')
    parser.add_argument('-f', dest='filename', type=str, default="test_fracas.easyccg.parsed.txt",
                        help='file that contains parsed premises, e.g. fracas.easyccg.parsed.txt / '
                             'fracas.candc.parsed.xml')
    args = parser.parse_args()

    if args.test:
        print('in test')
        test()
        exit()

    if args.sentNo == ['all']:
        args.sentNo = []

    # initialize trees
    trees = getMono.CCGtrees()

    if ".candc." in args.filename:
        parser = "candc"
        trees.readCandCxml(args.filename)  # ('tmp.candc.xml')
    elif ".easyccg." in args.filename:
        parser = "easyccg"
        trees.readEasyccgStr(args.filename)
    else:
        print("cannot determine parser from filename")
        exit()

    print("parser:", parser)

    # trees.readEasyccgStr('fracas.easyccg.parsed.txt')
    # trees.readEasyccgStr('test_fracas.easyccg.parsed.txt')

    # ----------------------------------
    # build knowledge
    print('-'*80)
    knowledge = Knowledge()
    # k.build_all()   # for FraCaS we don't need this
    knowledge.build_quantifier()

    for t_idx, t in trees.trees.items():
        # fix tree
        t.fixQuantifier()
        t.fixNot()
        if parser == 'candc':
            t.fixRC()  # only fix RC for candc

        knowledge.update_sent_pattern(t)  # patterns like: every X is NP
        knowledge.update_word_lists(t)    # nouns, subSecAdj, etc.

    knowledge.update_modifier()                  # adj + n < n, n + RC/PP < n, v + PP < v
    # knowledge.update_rules()             # if N < VP and N < N_1, then N < N_1 who VP
    knowledge.print_knowledge()

    # ----------------------------------
    # polarize
    print('-' * 80)
    t = trees.trees[5]

    if args.verbose in [0, 3]:
        t.printTree()

    t.mark()
    if args.verbose in [1, 3]:
        t.printTree()

    t.polarize()
    if args.verbose in [2, 3]:
        t.printTree()

    t.getImpSign()

    t.printSent()

    # ----------------------------------
    # replacement
    print('-'*80)
    print('NOW REPLACEMENT\n')
    t.replacement(k=knowledge)
    for i in t.inferences:
        i.replacement(k=knowledge)
        # for j in i.inferences:
        #     j.replacement(k=k)

    print('\nwe can infer:')
    t.printAllInferences()


def test():
    trees = getMono.CCGtrees()
    trees.readEasyccgStr('test_fracas.easyccg.parsed.txt')
    # trees.readEasyccgStr('fracas.easyccg.parsed.txt')

    # build knowledge
    knowledge = Knowledge()
    knowledge.build_quantifier()
    knowledge.build_morph_tense()

    for t_idx, t in trees.trees.items():
        knowledge.update_sent_pattern(t)  # patterns like: every X is NP
        knowledge.update_word_lists(t)    # nouns, subSecAdj, etc.
    knowledge.update_modifier()                  # adj + n < n, n + RC/PP < n, v + PP < v
    # knowledge.update_rules()             # if N < VP and N < N_1, then N < N_1 who VP

    knowledge.print_knowledge()


class SentenceBase:
    def __init__(self, gen_inf=False):
        self.Ps = []                  # list of wholeStrs
        self.H = None                 # one wholeStr
        self.Ps_ccgtree = []          # list of CCGtree
        self.H_ccgtree = None         # CCGtree
        self.inferences = set()       # all inferences that have been tried: set of wholeStrs
                                          # only as storage
        self.inferences_tree = []     # all inferences as a set of ccgtree

        self.fringe = []              # fringe as a list
        self.fringe_str_set = set()   # fringe as a set of wholeStrs
        self.k = None                 # knowledge base

        # self.fringe_contra = []
        # self.fringe_contra_str_set = set()
        self.contras_str = set()          # set of wholeStrs

        self.fringe_neutral = []
        self.fringe_neutral_str_set = set()
        self.neutrals_tree = []       # all neutrals
        self.neutrals_str_set = set() # all neutrals, wholeStr

        # if gen_inf: only generate infs and contras, do not solve the problems
        self.gen_inf = gen_inf

    def add_P_str(self, ccgtree):
        self.Ps.append(ccgtree.wholeStr)
        self.Ps_ccgtree.append(ccgtree)

    def add_H_str(self, ccgtree):
        self.H = ccgtree.wholeStr
        self.H_ccgtree = ccgtree

    def add_H_str_there_be(self, ccgtree):
        # situation 1:
        # ``there be N who VP'' --> ``some N VP''
        # e.g. There was an Italian who became the world's greatest tenor.
        # TODO
        if any([ccgtree.wholeStr.startswith(x) for x in THERE_BE]):
            if ' WHO ' in ccgtree.wholeStr:
                print("\n*** deal with there be ***\n")
                print("'there be' with 'who' in H!")
                print("old H str:", ccgtree.wholeStr)
                word_list = ccgtree.wholeStr.split(' ')
                idx_who = word_list.index('WHO')
                # check the word after be is a quantifier or not
                next_word = word_list[2]
                if next_word in QUANTIFIERS:
                    N_str = ' '.join(word_list[3:idx_who])
                else:
                    N_str = ' '.join(word_list[2:idx_who])
                VP_str = ' '.join(word_list[(idx_who+1):])
                new_H_str = 'SOME ' + N_str + ' ' + VP_str
                print("new H str:", new_H_str.strip())
                self.H = new_H_str.strip()    # only changed wholeStr
                self.H_ccgtree = ccgtree      # tree not changed
                return

            # situation 2: no 'who'
            # ``There are few committee members from Portugal'' -->
            # few committee members are from Portugal TODO
            else:  # has ``there be'', but no ``who''

                pass

        # otherwise
        self.H = ccgtree.wholeStr
        self.H_ccgtree = ccgtree

    def store_tried_inf(self, ccgtree):
        """ store tried inferences in self.inference, self.inference_tree """
        if ccgtree.wholeStr not in self.inferences:
            self.inferences.add(ccgtree.wholeStr)
            self.inferences_tree.append(ccgtree)

    # def add_inf_str(self, ccgtree):
    #     self.inferences.add(ccgtree.wholeStr)

    # def update_inf_str(self, list_inf_str):
    #     self.inferences.update(list_inf_str)

    def add_to_contras(self, ccgtree_str_list):
        for ccgtree_str in ccgtree_str_list:
            if ccgtree_str not in self.contras_str:
                self.contras_str.add(ccgtree_str)

    def add_to_fringe(self, ccgtree):
        """ add a ccgtree to fringe """
        if ccgtree.wholeStr not in self.fringe_str_set:
            self.fringe.append(ccgtree)
            self.fringe_str_set.add(ccgtree.wholeStr)

    def add_to_fringe_neutral(self, ccgtree):
        """ add a ccgtree to fringe_neutral """
        if ccgtree.wholeStr not in self.fringe_neutral_str_set:
            self.fringe_neutral.append(ccgtree)
            self.fringe_neutral_str_set.add(ccgtree.wholeStr)

    def add_to_neutral(self, ccgtree):
        """ add a ccgtree to neutral """
        if ccgtree.wholeStr not in self.neutrals_str_set:
            self.neutrals_tree.append(ccgtree)
            self.neutrals_str_set.add(ccgtree.wholeStr)

    # entailment search
    def solve_ids(self, depth_max, fracas_id):
        """ iterative deepening search
        0: entail, 1: contra, 2: unknown
        """
        solution = 2
        for depth in range(1, depth_max+1):
            eprint('\n* depth: ', depth)
            self.fringe = []
            self.fringe_str_set = set()
            solution = self.solve_ids_helper(depth, fracas_id)
            if solution == 0:
                break

        # todo: neutrals
        if self.gen_inf:
            for P in self.Ps_ccgtree:  # todo
                self.fringe_neutral.extend(P.replacement_neutral(self.k))

            while len(self.fringe_neutral) > 0:
                # removed is a CCGtree
                removed = self.fringe_neutral.pop(-1)  # dfs: removes the last item
                self.add_to_neutral(removed)
                neutrals = removed.replacement_neutral(self.k)
                for neutral in neutrals:
                    self.add_to_neutral(neutral)
                    neutral.neutral_depth = removed.neutral_depth + 1
                    if neutral.neutral_depth < depth_max:
                        self.add_to_fringe_neutral(neutral)

        if solution == 0: return "E"  # entail
        elif solution == 1: return "C"  # contra
        return "U"  # unknown

    def solve_ids_helper(self, depth_max, fracas_id):
        # ids algorithm
        # add all Ps
        for P in self.Ps_ccgtree:
            if not self.gen_inf:  # solve the problem
                if utils.similar_str(self.H, P.wholeStr):  # our goal! inf is a CCGtree
                    return 0  # "entail"
                elif utils.similar_str_contra(self.H, P.wholeStr):
                    return 1  # "contra"
            self.add_to_fringe(ccgtree=P)

        # add inferences from DET rule to fringe
        # this gets us: 026, 027
        if fracas_id in ['020', '026', '027']:
            infs_DET_rule = self.k.update_DET_rule()
            if infs_DET_rule:
                for inf in infs_DET_rule: self.fringe.append(inf)
            # NEW DET RULE from LogicalSystem
            LS = LogicalSystem(self.k)
            LS.manual()
            for P in self.Ps_ccgtree:
                LS.add_logical_sent(P)
                for inf in LS.compute_DET_rule():
                    self.add_to_fringe(inf)

        # print(LS)

        while len(self.fringe) > 0:
            # removed is a CCGtree
            removed = self.fringe.pop(-1)             # dfs: removes the last item
            # TODO infs and contras
            inferences, contras_trees = removed.replacement(self.k, self.gen_inf)  # explore: do replacement

            inferences.extend(removed.transform_RC2JJ())
            inferences.extend(removed.transform_JJ2RC())

            # get contradiction
            self.add_to_contras(removed.replacement_contra())
            self.add_to_contras((t.wholeStr for t in contras_trees))  # generator

            # print('\nremoved.inf_d:', removed.inf_depth)
            # print('num inf:', len(removed.inferences))
            # print('len fri:', len(self.fringe))
            for inf in inferences:  # inf is ccgtree, inferences = all successors
                self.store_tried_inf(inf)             # store tried inferences
                inf.inf_depth = removed.inf_depth+1   # my depth = my parent + 1
                if not self.gen_inf:
                    if utils.similar_str(self.H, inf.wholeStr):  # our goal! inf is a CCGtree
                        return 0                            # "entail"
                    # TODO
                    if utils.similar_str_contra(self.H, inf.wholeStr):
                        return 1                            # "contra"

                # update_modifier knowledge?? TODO
                if inf.inf_depth < depth_max:
                    self.add_to_fringe(inf)

            removed.inferences = []                 # clear its inferences, for next depth

        if self.gen_inf: return None                # only generating pairs, don't need a decision
        for contra_str in self.contras_str:
            if utils.similar_str(contra_str, self.H):
                return 1                            # "contra"
        return 2                                    # "unknown"

class Knowledge:
    def __init__(self):
        # frags: {string : Fragment} where we have Fragment.small and Fragment.big
        self.frags = {}  # TODO a word may have different types!
        self.numPairs = 0
        # for the following: { wholeStr : {'type': lNode/NtNode, 'type': lNode/NtNode} }
        # one word can have different types! play: (S[ng]\NP)/PP, or S[ng]\NP
        # e.g. { PLAY : {'(S\NP)/PP': lNode/NtNode, 'S\NP': lNode/NtNode} }
        self.nouns = {}      # a dict of wordAsStr(upper) : LeafNode; ONLY 'N' here, no 'NP'
        self.subsecAdj = {}  # a dict of LeafNode
        self.VPs = {}        # a dict of VP: NtNode
        self.NPs = {}        # a dict of NP: NtNode
        self.RCs = {}        # a dict of RC: NtNode
        self.PP4N = {}       # a dict of PP that modifies nouns: NtNode
        self.PP4V = {}       # a dict of PP that modifies verbs (adverbials): NtNode
        self.advs = {}       # adverbs
        self.verbs = {}      # verbs
        self.CDs = {}        # cardinals
        # quantifier relations: most, many, a few = several,
        # 'quantifier A B' relations where quantifier is not 'all'
        self.quant_rel = { "MOST" : set(), "MANY" : set(), "SEVERAL" : set() }
        # each set contains strings: A_B
        self.quant_rel_str = { "MOST" : set(), "MANY" : set(), "SEVERAL" : set() }

    def print_knowledge(self):
        print('#-------- begin: Knowledge --------#')
        for key in self.frags.keys():
            frag = self.frags[key]
            print(frag, frag.ccgtree.root.cat.semCat, frag.ccgtree.root.cat)  # Fragment
            print('\tsmall:', self.frags[key].small)
            print('\tbig:  ', self.frags[key].big)
            print('\tequal:', self.frags[key].equal)
            print('\tant:', self.frags[key].ant)
            print()
        print("nouns\n", "\n".join(["\t{}: {}".format(x_wholeStr, sorted(self.nouns[x_wholeStr].keys())) \
                         for x_wholeStr in self.nouns]))
        print("subsecAdj\n", "\n".join(["\t{}: {}".format(x_wholeStr, sorted(self.subsecAdj[x_wholeStr].keys())) \
                         for x_wholeStr in self.subsecAdj]))
        print("VPs\n", "\n".join(["\t{}: {}".format(x_wholeStr, sorted(self.VPs[x_wholeStr].keys())) \
                         for x_wholeStr in self.VPs]))
        print("NPs\n", "\n".join(["\t{}: {}".format(x_wholeStr, sorted(self.NPs[x_wholeStr].keys())) \
                         for x_wholeStr in self.NPs]))
        print("RCs\n", "\n".join(["\t{}: {}".format(x_wholeStr, sorted(self.RCs[x_wholeStr].keys())) \
                         for x_wholeStr in self.RCs]))
        print("PP4N\n", "\n".join(["\t{}: {}".format(x_wholeStr, sorted(self.PP4N[x_wholeStr].keys())) \
                         for x_wholeStr in self.PP4N]))
        print("PP4V\n", "\n".join(["\t{}: {}".format(x_wholeStr, sorted(self.PP4V[x_wholeStr].keys())) \
                         for x_wholeStr in self.PP4V]))
        print("advs\n", "\n".join(["\t{}: {}".format(x_wholeStr, sorted(self.advs[x_wholeStr].keys())) \
                         for x_wholeStr in self.advs]))
        print("verbs\n", "\n".join(["\t{}: {}".format(x_wholeStr, sorted(self.verbs[x_wholeStr].keys())) \
                         for x_wholeStr in self.verbs]))
        print("CDs\n", "\n".join(["\t{}: {}".format(x_wholeStr, sorted(self.CDs[x_wholeStr].keys())) \
                         for x_wholeStr in self.CDs]))
        print("\nquant_rel:")
        for quant, rels in self.quant_rel.items():
            print('* {}'.format(quant), end=':\n')
            for rel in rels:
                print('\t{} -- {} -- {}'.format(quant, rel[0].wholeStr, rel[1].wholeStr))
        print('#-------- end: Knowledge --------#')

    def add2WordList(self, node, wordList):
        """ add a node to wordlist if not already in there """
        if node.wholeStr not in wordList:
            # print("adding to KB: {}".format(node.wholeStr))
            wordList[node.wholeStr] = { node.cat.typeWOfeats : node}
        else:
            # 'play' may have multiple types: (S\NP)/PP, S\NP
            # already have the type
            if node.cat.typeWOfeats in wordList[node.wholeStr]: pass
            else:  # don't have the type yet!
                # print("adding to KB: {}".format(node.wholeStr))
                wordList[node.wholeStr][node.cat.typeWOfeats] = node

    def add_equal_pair(self, frag1_node, frag2_node, verbose=False):
        """ all = each = every. frag1 and frag2 are CCGtree """
        try:
            assert isinstance(frag1_node, getMono.LeafNode) or isinstance(frag1_node, getMono.NonTermNode)
            assert isinstance(frag2_node, getMono.LeafNode) or isinstance(frag2_node, getMono.NonTermNode)
        except AssertionError as e:
            print("\n\nadd_equal_pair takes frag1_node, frag2_node\n\n")
            print(e)
            exit()

        if verbose:
            eprint("adding to KB the pair: {} = {}".format(frag1_node.wholeStr, frag2_node.wholeStr))

        # transform to tree
        frag1_tree = getMono.CCGtree(TermNode=deepcopy(frag1_node)) if hasattr(frag1_node, 'pos') else getMono.CCGtree(
            NonTermNode=deepcopy(frag1_node))
        frag2_tree = getMono.CCGtree(TermNode=deepcopy(frag2_node)) if hasattr(frag2_node, 'pos') else getMono.CCGtree(
            NonTermNode=deepcopy(frag2_node))

        # add frag1 to self.frags[frag2.wholeStr].equal
        if frag1_tree.wholeStr not in self.frags:
            frag1AsFrag = Fragment(frag1_tree)
            frag1AsFrag.equal.append(Fragment(frag2_tree))
            self.frags[frag1_tree.wholeStr] = frag1AsFrag
        else:  # frag1.wholeStr in self.frags.keys():
            if frag2_tree.wholeStr not in [x.wholeStr for x in self.frags[frag1_tree.wholeStr].equal]:
                self.frags[frag1_tree.wholeStr].equal.append(Fragment(frag2_tree))

        # add frag2 to self.frags[frag1.wholeStr].equal
        if frag2_tree.wholeStr not in self.frags:
            frag2AsFrag = Fragment(frag2_tree)
            frag2AsFrag.equal.append(Fragment(frag1_tree))
            self.frags[frag2_tree.wholeStr] = frag2AsFrag
        else:  # frag2.wholeStr in self.frags.keys():
            if frag1_tree.wholeStr not in [x.wholeStr for x in self.frags[frag2_tree.wholeStr].equal]:
                self.frags[frag2_tree.wholeStr].equal.append(Fragment(frag1_tree))

        self.numPairs += 1

    def add_pair(self, pair, verbose=False):
        """ pair is a tuple = (small, big), both are Node """
        try:
            assert isinstance(pair[0], getMono.LeafNode) or isinstance(pair[0], getMono.NonTermNode)
            assert isinstance(pair[1], getMono.LeafNode) or isinstance(pair[1], getMono.NonTermNode)
        except AssertionError as e:
            print("\n\nadd_pair takes (node1, node2)\n\n")
            print(e)
            exit()

        small = deepcopy(pair[0])  # a node
        big = deepcopy(pair[1])   # a node

        # transform to tree
        small = getMono.CCGtree(TermNode=small) if hasattr(small, 'pos') else getMono.CCGtree(NonTermNode=small)
        big = getMono.CCGtree(TermNode=big) if hasattr(big, 'pos') else getMono.CCGtree(NonTermNode=big)

        if verbose:
            eprint("adding to KB the pair: {} < {}".format(small.wholeStr, big.wholeStr))

        # add big to self.frags[small.wholeStr]
        if small.wholeStr not in self.frags.keys():
            smallAsFrag = Fragment(small)
            smallAsFrag.big.append(Fragment(big))
            # -------------------------
            # add small itself to small.big and small.small
            # smallAsFrag.big.append(smallAsFrag)
            # smallAsFrag.small.append(smallAsFrag)
            # -------------------------
            smallAsFrag.wholeStr = small.wholeStr
            self.frags[small.wholeStr] = smallAsFrag

        else:  # small.wholeStr in self.frags.keys():
            if big.wholeStr not in [x.wholeStr for x in self.frags[small.wholeStr].big]:
                self.frags[small.wholeStr].big.append(Fragment(big))

        # add small to self.frags[big.wholeStr]
        if big.wholeStr not in self.frags.keys():
            bigAsFrag = Fragment(big)
            bigAsFrag.small.append(Fragment(small))
            # -------------------------
            # add big itself to big.big and big.small
            # bigAsFrag.big.append(bigAsFrag)
            # bigAsFrag.small.append(bigAsFrag)
            # -------------------------
            bigAsFrag.wholeStr = big.wholeStr
            self.frags[big.wholeStr] = bigAsFrag

        else:  # big.wholeStr in self.frags.keys():
            if small.wholeStr not in [x.wholeStr for x in self.frags[big.wholeStr].small]:
                self.frags[big.wholeStr].small.append(Fragment(small))

        self.numPairs += 1

    def add_antonym_pair(self, frag1_node, frag2_node, verbose=False):
        """  small <-> big  """
        try:
            assert isinstance(frag1_node, getMono.LeafNode) or isinstance(frag1_node, getMono.NonTermNode)
            assert isinstance(frag2_node, getMono.LeafNode) or isinstance(frag2_node, getMono.NonTermNode)
        except AssertionError as e:
            print("\n\nadd_antonym_pair takes frag1_node, frag2_node\n\n")
            print(e)
            exit()

        if verbose:
            eprint("adding to KB the pair: {} <=> {}".format(frag1_node.wholeStr, frag2_node.wholeStr))

        # transform to tree
        frag1_tree = getMono.CCGtree(TermNode=deepcopy(frag1_node)) if hasattr(frag1_node, 'pos') else getMono.CCGtree(
            NonTermNode=deepcopy(frag1_node))
        frag2_tree = getMono.CCGtree(TermNode=deepcopy(frag2_node)) if hasattr(frag2_node, 'pos') else getMono.CCGtree(
            NonTermNode=deepcopy(frag2_node))

        # add frag1 to self.frags[frag2.wholeStr].ant
        if frag1_tree.wholeStr not in self.frags:
            frag1AsFrag = Fragment(frag1_tree)
            frag1AsFrag.ant.append(Fragment(frag2_tree))
            self.frags[frag1_tree.wholeStr] = frag1AsFrag
        else:  # frag1.wholeStr in self.frags.keys():
            if frag2_tree.wholeStr not in [x.wholeStr for x in self.frags[frag1_tree.wholeStr].ant]:
                self.frags[frag1_tree.wholeStr].ant.append(Fragment(frag2_tree))

        # add frag2 to self.frags[frag1.wholeStr].ant
        if frag2_tree.wholeStr not in self.frags:
            frag2AsFrag = Fragment(frag2_tree)
            frag2AsFrag.ant.append(Fragment(frag1_tree))
            self.frags[frag2_tree.wholeStr] = frag2AsFrag
        else:  # frag2.wholeStr in self.frags.keys():
            if frag1_tree.wholeStr not in [x.wholeStr for x in self.frags[frag2_tree.wholeStr].ant]:
                self.frags[frag2_tree.wholeStr].ant.append(Fragment(frag1_tree))

        self.numPairs += 1

    def add_alternation_pair(self, frag1_node, frag2_node, verbose=False):
        """  frag1 | frag2  """
        pass


    def add_quant_rel(self, quant, node1, node2):
        """ add `quant node1 node2' to self.quant_rel, e.g. `most dogs animals' """
        if node1.wholeStr + '_' + node2.wholeStr not in self.quant_rel_str[quant]:
            # add a tuple (node1, node2)
            self.quant_rel[quant].add( (deepcopy(node1), deepcopy(node2)) )
            # add a string node1_node2
            self.quant_rel_str[quant].add(node1.wholeStr + '_' + node2.wholeStr)

    def update_quant_rel(self, sent_ccgtree):
        """ update K: most/many/several A B """
        # test case fracas-026: most Europeans are resident in Europe
        # we want: self.quant_rel = {'MOST': set( (node_A, node_B), (), ... ) }
        # node_A = Europeans, node_B = resident in Europe
        # TODO the parse from easyccg is wrong!
        if sent_ccgtree.wholeStr.split(' ')[0] in {'MOST', 'MANY', 'SEVERAL'}:
            # match structure:
            # most    A           are        B
            # NP/N    N          (S\NP)/NP   NP_2
            # --------------     ----------------
            #      NP_1                S\NP  =  VP
            #
            if "BE" in sent_ccgtree.words:
                node_most = sent_ccgtree.getLeftMostLeaf(sent_ccgtree.root)
                try:
                    assert node_most.wholeStr == "MOST"
                    N, NP_2, VP = None, None, None
                    if node_most.sisters: N = node_most.sisters[0]
                    if node_most.parent.sisters: VP = node_most.parent.sisters[0]
                    if VP:
                        if VP.cat.typeWOfeats == r"S\NP":
                            if len(VP.children[0].children) == 0:
                                if VP.children[0].lemma.upper() == "BE":
                                    NP_2 = VP.children[1]
                            print('\nin update_quant_rel\n')
                            if N and NP_2:
                                self.add_quant_rel('MOST', N, NP_2)
                except AssertionError:
                    print("something wrong in update_quant_rel")

        # TODO so now I manually add: most (Europeans, resident in Europe)
        node_N_1 = getMono.LeafNode(depth=0, cat=getMono.Cat(r"N", word="Europeans"), chunk=None, entity=None, lemma="Europeans",
                             pos="NNS", span=None, start=None, word="Europeans",
                             impType=None, fixed=False)
        node_resident = getMono.LeafNode(depth=0, cat=getMono.Cat(r"N", word="resident"), chunk=None, entity=None, lemma="resident",
                             pos="NN", span=None, start=None, word="resident",
                             impType=None, fixed=False)
        node_in = getMono.LeafNode(depth=0, cat=getMono.Cat(r"(N\N)/NP", word="in"), chunk=None, entity=None, lemma="in",
                                    pos="IN", span=None, start=None, word="in",
                                    impType=None, fixed=False)
        node_europe = getMono.LeafNode(depth=0, cat=getMono.Cat("NP", word="Europe"), chunk=None, entity=None, lemma="Europe",
                                    pos="NNP", span=None, start=None, word="Europe",
                                    impType=None, fixed=False)
        node_in_europe = getMono.NonTermNode(depth=0, cat=getMono.Cat(r"N\N"), ruleType='fa')
        node_in_europe.children = [node_in, node_europe]
        node_in.parent, node_europe.parent = node_in_europe, node_in_europe
        node_in_europe.assignWholeStr()

        node_resident_in_europe = getMono.NonTermNode(depth=0, cat=getMono.Cat("N"), ruleType='ba')
        node_resident_in_europe.children = [node_resident, node_in_europe]
        node_resident.parent, node_in_europe.parent = node_resident_in_europe, node_resident_in_europe
        node_resident_in_europe.assignWholeStr()

        self.add_quant_rel('MOST', node_N_1, node_resident_in_europe)  # N=leafNode, NP_2=nonTermNode

    def update_DET_rule(self):
        """ DET rule: test case fracas-026
        DET  x  y        All  x  z
        -------------------------  DET
              DET x (y ^ z)
        Note: z is a verb

        return `DET x (y ^ z)' as a CCGtree to be added
           to SentenceBase.fringe

        e.g.
        most dogs are animals        all animals bite
        -----------------------------------------------
                    most dogs are animals who bite
        """
        print('-' * 80)
        print('** update_DET_rule **\n')
        ans = []
        for det, tuple_set in self.quant_rel.items():
            if tuple_set:
                for tup in tuple_set:

                    # step 1.  DET x y: most dogs animals
                    x, y = tup[0], tup[1]   # x, y are nodes
                    print('\t{} -- {} -- {}'.format(det, x.wholeStr, y.wholeStr))

                    # step 2.  find ``all x z''
                    if self.frags[x.wholeStr].big:
                        for z in self.frags[x.wholeStr].big:  # z is Fragment!
                            print('x: ', x.wholeStr, x.cat.typeWOfeats)  # EUROPEANS, N
                            print('y: ', y.wholeStr, y.cat.typeWOfeats)  # RESIDENT IN EUROPE, N
                            print('z: ', z.wholeStr, z.ccgtree.root.cat.typeWOfeats)  # PEOPLE, NP

                            # make copies
                            x_copy, y_copy, z_copy = deepcopy(x), deepcopy(y), deepcopy(z.ccgtree.root)
                            print(y_copy.cat.typeWOfeats, z_copy.cat.typeWOfeats)

                            # ***  CASE 1  ***
                            # both y, z are noun, then return:
                            # "DET x be z who be y" and "DET x be y who be z"
                            # x.type = N;   y.type = NP;   z.type = N
                            # ---------------------------------------------------
                            # example: x = Europeans/N, y = resident in Europe/N, z = people/NP
                            # input:  most x y, all x z
                            # return: most x be z who be y
                            # ---------------------------------------------------
                            if y_copy.cat.typeWOfeats.startswith("N") and \
                                    z_copy.cat.typeWOfeats.startswith("N"):
                                # !! need to build a new !!

                                # case 1: y, z are both of type N

                                # case 2: y type N, z type NP
                                # in our example: fracas-026, we have y=N, z=NP
                                if y_copy.cat.typeWOfeats == "N" and \
                                        z_copy.cat.typeWOfeats == "NP":

                                    new_tree = self.generate_DET_rule(det, x_copy, y_copy, z_copy)
                                    ans.append(new_tree)

                                    # *** SANITY CHECK ***
                                    # new_tree.printTree()
                                    # print(new_tree.wholeStr)
                                    # print(new_tree.root.children[1].wholeStr)
                                    # print(new_tree.root.children[1].children[1].wholeStr)
                                    # print("here")

                                # case 3: y type NP, z type N

                                # case 4: y type NP, z type NP

                            # ***  CASE 2  ***
                            # y is noun, z is verb, then return:
                            # DET x be y who z

                            # ***  CASE 3  ***
                            # both y, z are verb, do nothing

                            # ***  CASE 4  ***
                            # y = N, z = PP

                            elif y_copy.cat.typeWOfeats == "PP" and \
                                    z_copy.cat.typeWOfeats == "NP":
                                new_tree = self.generate_DET_rule(det, x_copy, y_copy, z_copy)
                                ans.append(new_tree)

                            pass
                    else:  # no z such that ``all x z'', so do nothing
                        pass

                pass
        return ans

    def generate_DET_rule(self, det, x_copy, y_copy, z_copy):
        # goal: most x be1 z who be2 y
        if y_copy.cat.typeWOfeats == "N":
            type_node_be2 = r"(S\NP)/NP"
        elif y_copy.cat.typeWOfeats == "PP":
            type_node_be2 = r"(S\NP)/PP"
        else:
            print("cannot handle y in generate_DET_rule")
            return None

        node_be2 = getMono.LeafNode(depth=0,
                                    cat=getMono.Cat(type_node_be2, word='be'),
                                    chunk=None, entity=None,
                                    lemma='be', pos='VBP', span=None,
                                    start=None, word='be')
        node_be1 = getMono.LeafNode(depth=0,
                                    cat=getMono.Cat(r"(S\NP)/NP", word='be'),
                                    chunk=None, entity=None,
                                    lemma='be', pos='VBP', span=None,
                                    start=None, word='be')
        node_det = getMono.LeafNode(depth=0, cat=getMono.Cat(r'NP/N', word=det.lower()),
                                    chunk=None, entity=None,
                                    lemma=det.lower(), pos='DT', span=None, start=None,
                                    word=det.lower())
        node_who = getMono.LeafNode(depth=0, cat=getMono.Cat(r'(N\N)/(S\NP)', word='who'),
                                    chunk=None, entity=None,
                                    lemma='who', pos='WP', span=None, start=None,
                                    word='who')
        if y_copy.cat.typeWOfeats == "N":
            node_y_NP = getMono.NonTermNode(depth=0, cat=getMono.Cat(r'NP'),
                                            ruleType='lex')
            node_y_NP.children = [y_copy]
            y_copy.parent = node_y_NP
        elif y_copy.cat.typeWOfeats == "PP":
            node_y_NP = y_copy
        else:
            print("cannot handle y in generate_DET_rule")
            return None

        node_be2_y = getMono.NonTermNode(depth=0, cat=getMono.Cat(r'S\NP'), ruleType='fa')
        node_be2_y.children = [node_be2, node_y_NP]
        node_be2.parent, node_y_NP.parent = node_be2_y, node_be2_y

        node_who_be2_y = getMono.NonTermNode(depth=0, cat=getMono.Cat(r'N\N'), ruleType='fa')
        node_who_be2_y.children = [node_who, node_be2_y]
        node_who.parent, node_be2_y.parent = node_who_be2_y, node_who_be2_y

        # need another lex rule, but modified: N\N -> NP\NP
        # node_who_be2_y_NP = getMono.NonTermNode(depth=0, cat=getMono.Cat(r'NP\NP'), ruleType='lex')
        # node_who_be2_y_NP.children = [node_who_be2_y]
        # node_who_be2_y.parent = node_who_be2_y_NP

        # z = NP
        # unlex z from NP to N
        node_z_N = getMono.NonTermNode(depth=0, cat=getMono.Cat(r'N'), ruleType='unlex')
        node_z_N.children = [z_copy]
        z_copy.parent = node_z_N

        node_z_who_be2_y = getMono.NonTermNode(depth=0, cat=getMono.Cat(r'N'), ruleType='ba')
        node_z_who_be2_y.children = [node_z_N, node_who_be2_y]
        node_z_N.parent, node_who_be2_y.parent = node_z_who_be2_y, node_z_who_be2_y

        # another lex
        node_z_who_be2_y_NP = getMono.NonTermNode(depth=0, cat=getMono.Cat(r'NP'), ruleType='lex')
        node_z_who_be2_y_NP.children = [node_z_who_be2_y]
        node_z_who_be2_y.parent = node_z_who_be2_y_NP

        node_be1_z_who_be2_y = getMono.NonTermNode(depth=0, cat=getMono.Cat(r'S\NP'), ruleType='fa')
        node_be1_z_who_be2_y.children = [node_be1, node_z_who_be2_y_NP]
        node_be1.parent, node_z_who_be2_y_NP.parent = node_be1_z_who_be2_y, node_be1_z_who_be2_y

        node_det_x = getMono.NonTermNode(depth=0, cat=getMono.Cat(r'NP'), ruleType='fa')
        node_det_x.children = [node_det, x_copy]
        node_det.parent, x_copy.parent = node_det_x, node_det_x

        node_sent = getMono.NonTermNode(depth=0, cat=getMono.Cat(r'S'), ruleType='ba')
        node_sent.children = [node_det_x, node_be1_z_who_be2_y]
        node_det_x.parent, node_be1_z_who_be2_y.parent = node_sent, node_sent

        new_tree = getMono.CCGtree(NonTermNode=node_sent)
        new_tree.mark()
        new_tree.polarize()

        print('adding to s.fringe:', end=' ')
        new_tree.printSent()
        return new_tree



    def update_sent_pattern(self, sent_ccgtree):
        """ every x y --> x <= y """
        # update_modifier knowledge base after reading a sentence(premise) if certain pattern matches
        if sent_ccgtree.wholeStr.split(' ')[0] in {'EVERY', 'ALL', 'EACH'}:
            # if sentence: "every X be Y", then add X < Y
            # TODO what if 2 IS in sentence?
            self.pattern_be(sent_ccgtree)

            # if sentence: "every Y VP", then add Y < VP
            # need to do this no matter "be" in sent or not
            # TODO this applies to every sentence starting with 'every'
            self.pattern_VP(sent_ccgtree)

    def pattern_be(self, tree):
        if "BE" in tree.words:
            # go to the node "is"
            for leaf_node in tree.leafNodes:
                if leaf_node.word.upper() == "BE" and leaf_node.cat.typeWOfeats == r"(S\NP)/NP":
                    X = None  # the head N inside NP_1
                    Y = None  # the head N inside NP_2
                    NP_2, NP_1 = None, None

                    # must match the following structure
                    #    every     X           be        Y
                    #    NP/N      N       (S\NP)/NP     NP_2
                    #   -------------      ------------------
                    #         NP_1               S\NP
                    #
                    node_be = leaf_node

                    # find Y in NP_2
                    if node_be.sisters:
                        if node_be.sisters[0].cat.typeWOfeats == "NP":
                            NP_2 = leaf_node.sisters[0]
                            if NP_2.children:  # make sure it's ISA relation
                                # i.e. every European is a person
                                if NP_2.children[0].wholeStr in {"A", "AN"}:
                                    Y = NP_2.children[1]
                                # i.e. all Europeans are people
                                else:
                                    Y = NP_2

                    # find X in NP_1
                    if node_be.parent.sisters:
                        NP_1 = node_be.parent.sisters[0]
                        if NP_1.cat.typeWOfeats == "NP":
                            if NP_1.children:
                                if NP_1.children[0].word.upper() in {'EVERY', 'ALL', 'EACH'}:
                                    if NP_1.children[1].cat.typeWOfeats == "N":
                                        X = NP_1.children[1]

                    if NP_1 and X and NP_2 and Y:
                        eprint("adding to Knowledge base(every X be Y):", X.wholeStr, "<", Y.wholeStr)
                        self.add_pair((X, Y))
                        # self.add_pair((
                        #     (getMono.CCGtree(NonTermNode=deepcopy(X)), (getMono.CCGtree(NonTermNode=deepcopy(Y))))
                        # ))
                        for equal in self.frags[X.wholeStr].equal:
                            eprint("adding to K (every X be Y):", X.wholeStr, "<", Y.wholeStr)
                            self.add_pair((equal.ccgtree.root, Y))
                            # self.add_pair((
                            #     (deepcopy(equal.ccgtree), (getMono.CCGtree(NonTermNode=deepcopy(Y))))))

    def pattern_VP(self, tree):
        node_every = tree.getLeftMostLeaf(tree.root)  # leftmost node
        try:
            assert node_every.wholeStr in {'EVERY', 'ALL', 'EACH'}
        except AssertionError:
            print("something wrong updating `every Y VP' to `Y < VP'")
            tree.printSent()
            return
        # must match the following structure
        #    every     Y
        #    NP/N      N
        #   -------------             VP
        #         NP                  S\NP

        Y = None  # the head N inside NP
        VP = None

        # find Y
        if node_every.sisters:
            Y = node_every.sisters[0]
            VP = node_every.parent.sisters[0]

            # make sure the VP is not a copula VP: be ...
            if VP.children:
                if VP.children[0].wholeStr == "BE":
                    pass
        if Y.cat.typeWOfeats == "N" and VP.cat.typeWOfeats == r"S\NP":
            # print("adding to Knowledge base(every Y VP):", Y.wholeStr, "<", VP.wholeStr)
            # self.add_pair(((getMono.CCGtree(NonTermNode=deepcopy(Y)),
            #                 (getMono.CCGtree(NonTermNode=deepcopy(VP))))))
            # for equal in self.frags[Y.wholeStr].equal:
            #     print("adding to Knowledge base(every Y VP):", Y.wholeStr, "<", VP.wholeStr)
            #     self.add_pair(((deepcopy(equal.ccgtree),
            #                     (getMono.CCGtree(NonTermNode=deepcopy(VP))))))

            # add: 'be Y' < 'VP'
            # Y type: N
            Y_copy = deepcopy(Y)
            node_Y_NP = getMono.NonTermNode(depth=0, cat=getMono.Cat(originalType=r'NP'), ruleType='lex')
            node_Y_NP.children = [Y_copy]
            Y_copy.parent = node_Y_NP
            node_Y_NP.assignWholeStr()

            node_be = getMono.LeafNode(depth=0,
                                       cat=getMono.Cat(r'(S\NP)/NP', word='be'),
                                       chunk=None, entity=None,
                                       lemma='be', pos='VBP', span=None,
                                       start=None, word='be')

            node_be_Y = getMono.NonTermNode(depth=0, cat=getMono.Cat(r'S\NP'), ruleType='fa')
            node_be_Y.children = [node_be, node_Y_NP]
            node_be.parent, node_Y_NP.parent = node_be_Y, node_be_Y
            node_be_Y.assignWholeStr()
            eprint("adding to Knowledge base(be Y < VP):", node_be_Y.wholeStr, "<", VP.wholeStr)
            self.add_pair((node_be_Y, VP))

    def update_word_lists(self, sent_ccgtree):
        """ update_modifier self.nouns, subsecAdj, VP ... when read in a sent
        self.nouns = {}      # a dict of wordAsStr(upper) : LeafNode; ONLY 'N' here, no 'NP'
        self.subsecAdj = {}  # a dict of LeafNode
        self.VPs = {}         # a dict of VP: NtNode
        self.NPs = {}         # a dict of NP: NtNode
        self.RCs = {}         # a dict of RC: NtNode
        self.PP4N = {}       # a dict of PP that modifies nouns: NtNode
        self.PP4V = {}       # a dict of PP that modifies verbs (adverbials): NtNode
        """
        # nouns, adjs
        for lnode in sent_ccgtree.leafNodes:
            if lnode.cat.typeWOfeats == "N":
                self.add2WordList(lnode, self.nouns)
            elif lnode.cat.typeWOfeats == r"N/N" and lnode.pos != "CD":
                if lnode.word.lower() not in NON_SUBSEC_ADJ:
                    self.add2WordList(lnode, self.subsecAdj)
            # RB: adverbs
            elif lnode.pos == "RB" and \
                    lnode.word.lower() not in {"as", "not"}:
                # print("adding adverb to KB:", lnode)
                self.add2WordList(lnode, self.advs)
            # verbs
            elif lnode.pos.startswith("VB"):
                # print("adding verb to KB:", lnode)
                self.add2WordList(lnode, self.verbs)

        # VP, NP, RC, PP4N, PP4V
        # N: non term node can also be N, e.g. musical instrument
        for ntnode in sent_ccgtree.nonTermNodes:
            try:
                # PP4V: (S\NP)\(S\NP)
                if ntnode.cat.typeWOfeats == r"(S\NP)\(S\NP)" and \
                        ntnode.children[0].pos == "IN":  # IN: preposition
                    self.add2WordList(ntnode, self.PP4V)

                # PP4N: N\N (easyccg) or NP\NP (candc)
                elif ntnode.cat.typeWOfeats in {r"N\N", r"NP\NP"} and \
                        ntnode.children[0].pos == "IN":  # IN: preposition
                    self.add2WordList(ntnode, self.PP4N)

                # RC: N\N (easyccg) or NP\NP (candc)
                elif ntnode.cat.typeWOfeats in {r"N\N", r"NP\NP"} and \
                        ntnode.children[0].word.lower() in {"who", "that", "which", "whom"}:
                    self.add2WordList(ntnode, self.RCs)

                # VP: S\NP
                elif ntnode.cat.typeWOfeats == r"S\NP":
                    self.add2WordList(ntnode, self.VPs)

                # N
                elif ntnode.cat.typeWOfeats == r"N":
                    self.add2WordList(ntnode, self.nouns)

                # NP
                elif ntnode.cat.typeWOfeats == 'NP':
                    self.add2WordList(ntnode, self.NPs)

            except AttributeError:
                pass

    def update_modifier(self):
        """  # adj + n < n, n + RC/PP < n, v + PP < v """
        self.modifier_NP()  # adj + n < n; n + RC/PP < n;
        self.modifier_VP()  # v + PP < v
        self.modifier_JJ()  # very + adj
        self.modifier_PP()  # not implemented

    def modifier_PP(self):
        """ very (near a red barrel) < near a red barrel """
        pass

    def modifier_NP(self):
        # 1. adj + n < n
        if self.nouns and self.subsecAdj:
            for noun_dict in self.nouns.values():
                for noun in noun_dict.values():    # noun is a LeafNode / NonTermNode
                    # make sure that noun does not already have an adj:
                    try:  # todo: is this working?
                        if len(noun.children) == 2 and noun.children[0].pos == "JJ": # good dog
                            continue
                    except AttributeError: pass
                    for adj_dict in self.subsecAdj.values():
                        for adj in adj_dict.values():
                            # print("!!! here")
                            # print(noun)
                            # print(adj)
                            # new_noun is the mother of adj + noun
                            new_noun = getMono.NonTermNode(depth=0,
                                                           cat=getMono.Cat(originalType='N',
                                                                           word=None),
                                                           ruleType='fa')
                            adj_copy = deepcopy(adj)
                            noun_copy = deepcopy(noun)
                            self.modifier_NP_helper(new_noun, adj_copy, noun_copy, "ADJ", True)

        # 2. n + RC < n;
        if self.nouns and self.RCs:
            for noun_dict in self.nouns.values():  # noun is a LeafNode
                for noun in noun_dict.values():  # noun is a LeafNode
                    for RC_dict in self.RCs.values():
                        for RC in RC_dict.values():
                            new_noun = getMono.NonTermNode(depth=0,
                                                           cat=getMono.Cat(originalType='N',
                                                                           word=None),
                                                           ruleType='ba')
                            RC_copy = deepcopy(RC)
                            noun_copy = deepcopy(noun)
                            self.modifier_NP_helper(new_noun, RC_copy, noun_copy, "RC", True)

        # 3. n + PP4N < n
        if self.nouns and self.PP4N:
            for noun_dict in self.nouns.values():  # noun is a LeafNode
                for noun in noun_dict.values():  # noun is a LeafNode
                    for PP_dict in self.PP4N.values():
                        for PP in PP_dict.values():
                            new_noun = getMono.NonTermNode(depth=0,
                                                           cat=getMono.Cat(originalType='N',
                                                                           word=None),
                                                           ruleType='ba')
                            PP_copy = deepcopy(PP)
                            noun_copy = deepcopy(noun)
                            self.modifier_NP_helper(new_noun, PP_copy, noun_copy, "PP")

    def modifier_NP_helper(self, new_noun, modifier_copy, noun_copy, relation, verbose=False):
        """ modifier can be: 1. adj, which is pre-nominal
        2. PP and RC, which are post-nominal """
        if relation in ["PP", "RC"]:
            self.set_parent_children(new_noun, [noun_copy, modifier_copy])
            # new_noun.children = [noun_copy, modifier_copy]
        elif relation in ["ADJ"]:
            self.set_parent_children(new_noun, [modifier_copy, noun_copy])
            # new_noun.children = [modifier_copy, noun_copy]
        # set parent and sister relations
        # modifier_copy.parent = new_noun
        # noun_copy.parent = new_noun
        # modifier_copy.sisters = [noun_copy]
        # noun_copy.sisters = [modifier_copy]
        # new_noun.assignWholeStr()
        # print("!!here")
        # print(new_noun.wholeStr)
        # print(noun_copy)
        self.add_pair((new_noun, noun_copy), verbose)

    def modifier_VP(self):
        # 4. VP + PP4V < VP
        if self.VPs and self.PP4V:
            for VP_dict in self.VPs.values():  # VP_dict = {'(S\NP)/PP': lNode/NtNode, 'S\NP': lNode/NtNode} for PLAY
                for VP in VP_dict.values():  # VP = ntNode
                    for PP_dict in self.PP4V.values():
                        for PP in PP_dict.values():
                            self.modifier_VP_post_mod(VP, PP)

        # 5. adv + vp/v < vp/v, vp/v + adv < vp/v
        if self.advs:
            for adv_dict in self.advs.values():
                for adv in adv_dict.values():
                    if adv.cat.typeWOfeats == r'(S\NP)\(S\NP)':

                        # adv + v < v
                        if self.verbs:
                            for verb_dict in self.verbs.values():
                                for verb in verb_dict.values():
                                    # print(verb.cat.typeWOfeats, verb)
                                    # case 1: leap fearlessly: (S\NP) (S\NP)\(S\NP) ba -> (S\NP)
                                    if verb.cat.typeWOfeats == r'S\NP':
                                        self.modifier_VP_post_mod(verb, adv)
                                    # case 2: be carefully (doing sth): (S\NP)/(S\NP) (S\NP)\(S\NP) bx -> (S\NP)/(S\NP)
                                    elif verb.cat.typeWOfeats == r'(S\NP)/(S\NP)' and \
                                            verb.word.lower() == 'be':
                                        self.modifier_VP_post_mod(verb, adv, 'bx')
                                    # case 3: transitive verb: (S\NP)/NP ?? TODO

                        # VP + adv < VP : VP type = S\NP # TODO will this over-generalize?
                        if self.VPs:
                            for VP_dict in self.VPs.values():
                                for VP in VP_dict.values():
                                    self.modifier_VP_post_mod(VP, adv)

                    elif adv.cat.typeWOfeats == r'(S\NP)/(S\NP)':  # pre-VP mod
                        # adv + VP < VP
                        if self.VPs:
                            for VP_dict in self.VPs.values():
                                for VP in VP_dict.values():
                                    self.modifier_VP_pre_mod(VP, adv)

        # verb PP < verb: play in a lovely way < play
        if self.VPs:
            for VP_dict in self.VPs.values():
                for VP in VP_dict.values():
                    if VP.children and (len(VP.children[0].children) == 0) and \
                            (VP.children[1].cat.typeWOfeats == "PP"):
                        if VP.children[0].pos.startswith("VB"):
                            # the type of verb must be the same as VP
                            type_VP = VP.cat.typeWOfeats
                            verb_old = VP.children[0]
                            verb_new = getMono.LeafNode(depth=0, cat=getMono.Cat(originalType=type_VP, word=verb_old.word),
                                                        chunk=None, entity=None, lemma=verb_old.lemma,
                                                        pos="VBZ", span=None, start=None,
                                                        word=verb_old.word)
                            self.add_pair((VP, verb_new), True)

    def modifier_VP_post_mod(self, old_VP, post_VP_modifier, ruletype='ba'):
        """ VP + PP4V < VP; VP + adv < VP """
        new_vp_type = r'S\NP'
        if ruletype == 'bx': new_vp_type = r'(S\NP)/(S\NP)'
        new_vp = getMono.NonTermNode(depth=0,
                                     cat=getMono.Cat(originalType=new_vp_type,word=None),
                                     ruleType=ruletype)
        VP_copy = deepcopy(old_VP)
        modifier_copy = deepcopy(post_VP_modifier)

        self.set_parent_children(new_vp, [VP_copy, modifier_copy])
        self.add_pair((new_vp, VP_copy), True)

    def modifier_VP_pre_mod(self, old_VP, pre_VP_modifier, ruletype='fa'):
        """ adv + VP < VP : recklessly climb a rope=new_vp < climb a rope=VP_copy """
        new_vp_type = r'S\NP'
        # if ruletype == 'bx': new_vp_type = r'(S\NP)/(S\NP)'
        new_vp = getMono.NonTermNode(depth=0,
                                     cat=getMono.Cat(originalType=new_vp_type,word=None),
                                     ruleType=ruletype)
        VP_copy = deepcopy(old_VP)
        modifier_copy = deepcopy(pre_VP_modifier)

        self.set_parent_children(new_vp, [modifier_copy, VP_copy])
        self.add_pair((new_vp, VP_copy), True)

    def modifier_JJ(self):
        """ very sunny < sunny: SICK 6315 """
        for adj_dict in self.subsecAdj.values():
            for adj in adj_dict.values():
                if adj.pos == "CD": continue  # exclude: "very 8"
                # exclude "very very nice"
                try:
                    if len(adj.children) == 2 and adj.children[0].word == "very": continue
                except AttributeError: pass
                # new_adj is (very + JJ)
                adj_type = adj.cat.typeWOfeats
                if adj_type == r'N/N':
                    very_node = utils.make_leafnode(cat=r'(N/N)/(N/N)', word='very', lemma='very', pos='RB')
                else:
                    very_node = utils.make_leafnode(cat=r'('+adj_type+')/('+adj_type+')', word='very', lemma='very', pos='RB')
                new_adj = getMono.NonTermNode(depth=0,
                                              cat=getMono.Cat(originalType=r'N/N',
                                                            word=None),
                                              ruleType='fa')
                adj_copy = deepcopy(adj)
                # set parent and sister relations
                self.set_parent_children(new_adj, [very_node, adj_copy])
                self.add_pair((new_adj, adj_copy), True)

    def set_parent_children(self, parent_node, children_list):
        """ set relations between parent and children,
        parent: a node, children: a list """
        parent_node.children = children_list
        for child in children_list: child.parent = parent_node
        if len(children_list) == 2:
            children_list[0].sisters = [children_list[1]]
            children_list[1].sisters = [children_list[0]]
        parent_node.assignWholeStr()

    r"""
    def build_modifier(self, node, modifier, new_node):
        # cannot work on noun / adj directly
        node_copy = deepcopy(node)
        modifier_copy = deepcopy(modifier)

        new_node.children = [node_copy, modifier_copy]
        modifier_copy.parent = new_node
        node_copy.parent = new_node
        modifier_copy.sisters = [node_copy]
        node_copy.sisters = [modifier_copy]
        new_node.assignWholeStr()
        self.add_pair((getMono.CCGtree(NonTermNode=new_node),
                       getMono.CCGtree(TermNode=node_copy)))
    """

    r"""
    def update_rules(self):
        # update_modifier K based on the K-updating rules
        # since we are updating self.frags, we cannot loop over it.
        # Need a copy to loop over
        frags_tmp = deepcopy(self.frags)

        # ---------------------------------------
        # rule 1: if N < VP and N < N_1, then N < N_1 who VP
        # ---------------------------------------
        for frag in frags_tmp.values():
            # only need N
            if frag.ccgtree.root.cat.typeWOfeats == "N":
                Ns, VPs = [], []
                for big in frag.big:  # big is an object of Fragment
                    # find VP and N_1
                    if big.ccgtree.root.cat.typeWOfeats == r"S\NP":
                        VPs.append(big)
                    if big.ccgtree.root.cat.typeWOfeats == r"N":
                        Ns.append(big)

                # now we know that frag (which is an N) is smaller than another N and a VP
                # add N < N_1 who VP
                if Ns and VPs:  # if both not empty
                    self.build_N_VP_rule(frag, Ns, VPs)

    def build_N_VP_rule(self, frag, Ns, VPs):
        for N in Ns:
            for VP in VPs:
                # node_N and node_VP are Fragment! First convert them back to node
                node_N = deepcopy(N.ccgtree.root)  # LeafNode
                node_VP = deepcopy(VP.ccgtree.root)  # NonTermNode

                # N who VP
                node_who = getMono.LeafNode(depth=0,
                                            cat=getMono.Cat(r'(N\N)/(S\NP)', word='who'),
                                            chunk=None, entity=None,
                                            lemma='who', pos='WP', span=None, start=None,
                                            word='who')

                node_who_VP = getMono.NonTermNode(depth=0, cat=getMono.Cat(r'S\NP'),
                                                  ruleType="fa",
                                                  wholeStr='WHO ' + node_VP.wholeStr,
                                                  # wholeStrLemma='WHO '+node_VP.wholeStrLemma,
                                                  impType=None)
                # set up children, parent, sisters
                node_who_VP.children = [node_who, node_VP]
                node_who.parent = node_who_VP
                node_VP.parent = node_who_VP
                node_who.sisters = [node_VP]
                node_VP.sisters = [node_who]

                node_who_VP.assignWholeStr()

                node_N_who_VP = getMono.NonTermNode(depth=0, cat=getMono.Cat(r'N'),
                                                    ruleType="ba",
                                                    wholeStr=node_N.wholeStr + ' WHO ' + VP.wholeStr,
                                                    # wholeStrLemma=node_N.wholeStrLemma + ' WHO ' + VP.wholeStrLemma,
                                                    impType=None)
                # set up children, parent, sisters
                node_N_who_VP.children = [node_N, node_who_VP]
                node_N.parent = node_N_who_VP
                node_who_VP.parent = node_N_who_VP
                node_N.sisters = [node_who_VP]
                node_who_VP.sisters = [node_N]

                node_N_who_VP.assignWholeStr()

                # !!!  add  N < N_1 who VP  !!!
                self.add_pair((deepcopy(frag.ccgtree), getMono.CCGtree(NonTermNode=node_N_who_VP)))
                print('adding to Knowledge base(N < N_1 who VP):',
                      frag.ccgtree.wholeStr, '<', node_N_who_VP.wholeStr)

                # also add frag.equal
                # print(frag.equal)
                for equal in frag.equal:
                    self.add_pair((deepcopy(equal.ccgtree), getMono.CCGtree(NonTermNode=node_N_who_VP)))
                    print('adding to Knowledge base(N < N_1 who VP):',
                          frag.ccgtree.wholeStr, '<', node_N_who_VP.wholeStr)
    """

    r"""
    def build_all(self):
        '''
        June 2018

        We have 3 types of input for knowledge K:
        1. PAIRS: pairs in text format. e.g. N: dog < animal
        2. SUBSADJ: a list of subsective adjs. e.g. old, young
        3. ISA: isA sentences

        buildKnowledge():
        - nouns = []
        - build knowledge for each pair in PAIRS. (get nouns)
        - parse sentences in ISA. (get nouns)
          X is a Y.
          extract X and Y as treelets from the CCG parse tree
          then add the following to knowledge: every Y < X < some Y
        - find all the nouns in the system.
          nouns = nouns in PAIRS + nouns in ISA
          for every noun in nouns:
            for every adj in SUBSADJ:
              add this to knowledge: adj + noun < noun
        '''
        print('-' * 80)
        print('----------building knowledge...')
        self.build_pairs()
        self.build_ISA()
        self.build_SubsecAdj()
        self.build_quantifier()  # self defined relations
        print('\nknowledge built!\n--------------\n')

    def build_pairs(self):
        print('building knowledge from pairs ...\n')
        # read in pairs.txt in ./k
        with open('./k/pairs.txt') as f:
            for line in f:
                if line.startswith('#'):
                    continue

                syntactic_type = line.split(':')[0]  # N or V (or NP)

                if syntactic_type == 'Vt':
                    # make it None for now
                    # TODO but need to add this information in getMono.CCGtree.replacement()
                    syntactic_type = r'(S\NP)/NP'
                    pos = 'V'
                elif syntactic_type == 'Vi':
                    syntactic_type = r'S\NP'
                    pos = 'V'
                elif syntactic_type == 'N':
                    pos = 'N'
                else:
                    print('cannot handle syntactic type in pairs:', syntactic_type)
                    sys.exit(1)

                relation_pair = line.split(':')[1].split('<')  # [' dog ', ' animal ']
                try:
                    assert len(relation_pair) == 2
                except AssertionError:
                    print('wrong format in pairs.txt:')
                    print(line)

                # print(relationPair)

                # set small and big to LeafNode.
                small = getMono.LeafNode(depth=0,
                                         cat=getMono.Cat(originalType=syntactic_type,
                                                         word=relation_pair[0].strip()),
                                         chunk=None, entity=None, lemma=None,
                                         pos=pos, span=None, start=None,
                                         word=relation_pair[0].strip())
                big = getMono.LeafNode(depth=0,
                                       cat=getMono.Cat(originalType=syntactic_type,
                                                       word=relation_pair[1].strip()),
                                       chunk=None, entity=None, lemma=None,
                                       pos=pos, span=None, start=None,
                                       word=relation_pair[1].strip())

                # add N to self.nouns
                if syntactic_type == 'N':
                    for node in [small, big]:  # small and big are LeafNodes
                        if node.wholeStr not in self.nouns:
                            self.nouns[node.wholeStr] = node

                # change small and big to CCGtree().
                small = getMono.CCGtree(TermNode=small)
                big = getMono.CCGtree(TermNode=big)

                self.add_pair((small, big))
        print('...done!\n')
        # print(self.nouns)

    def build_SubsecAdj(self):
        # Read in adjs from k/subsecAdj.txt.
        print('reading subsective adjs from k/subsecAdj.txt...')
        # read in all subsec adjs
        with open('./k/subsecAdj.txt') as f:
            for line in f:
                if line.startswith('#'): continue
                adj = getMono.LeafNode(depth=0,
                                       cat=getMono.Cat(originalType=r'N/N',
                                                       word=line.strip()),
                                       chunk=None, entity=None, lemma=None,
                                       pos='JJ', span=None, start=None,
                                       word=line.strip())
                self.add2WordList(adj, self.subsecAdj)
        print('...done!\n')
    """

    r"""
    def build_ISA(self):
        # build knowledge from ./k/sentences4k.txt

        print('building knowledge from sentences4k.txt...')
        # parse ./k/sentences4k.txt
        print('parsing...')
        parseCommand = 'bash ./candcParse_visualize.sh ./k/sentences4k.txt k'
        os.system(parseCommand)

        # read from sentences4k.candc.xml
        # for each sentence, find the subject and object of ISA
        knowledgetrees = getMono.CCGtrees()
        knowledgetrees.readCandCxml('./k/sentences4k.candc.xml')
        print('\ntrees read in from sentences4k.candc.xml!\n')

        for idx, t in knowledgetrees.trees.items():
            t.fixQuantifier()
            t.fixNot()
            t.fixRC()
            t.mark()
            t.polarize()

            print('isA relation:', end=' ')
            t.printSent()
            print()

            subj, pred = t.getSubjPredISA()  # subj, pred are LeafNode or NonTermNode

            # if these is ISA relation
            if (subj is not None) and (pred is not None):
                # add N to self.nouns
                for node in [subj, pred]:
                    try:
                        if node.cat.typeWOfeats == 'N' and node.word.upper() not in self.nouns:
                            # print('adding', node, 'to nouns')
                            self.nouns[node.wholeStr.upper()] = node
                    except AttributeError:  # node is not a LeafNode, has no pos
                        pass

                # add to knowledge
                #   every dog  < John < some dog
                # = every pred < subj < some pred
                every = getMono.LeafNode(depth=0,cat=getMono.Cat('NP/N',word='every'),
                                         chunk=None,entity=None,
                                         lemma='every',pos='DT',span=None,start=None,
                                         word='every')

                some = getMono.LeafNode(depth=0,cat=getMono.Cat('NP/N',word='some'),
                                        chunk=None,entity=None,
                                        lemma='some',pos='DT',span=None,start=None,
                                        word='some')

                # initialize node for phrase [every dog], [some dog]
                everyDogNode = getMono.NonTermNode(depth=0,cat=getMono.Cat('NP'),ruleType='fa')
                someDogNode = getMono.NonTermNode(depth=0,cat=getMono.Cat('NP'),ruleType='fa')

                # fix children, parent, sister relations
                everyDogNode.children = [every, pred]
                every.parent = everyDogNode; pred.parent  = everyDogNode
                every.sisters = [pred]     ; pred.sisters = [every]

                predCopy = pred.copy()  # make a new copy for dog
                someDogNode.children = [some, predCopy]
                some.parent = someDogNode; predCopy.parent = someDogNode
                some.sisters = [predCopy]; predCopy.sisters = [some]

                # make them +
                everyDogNode.cat.semCat.marking = '+'
                someDogNode.cat.semCat.marking = '+'
                # print('everyDogNode:', everyDogNode)
                # print('someDogNode:', someDogNode)

                everyDogNode.assignWholeStr()
                someDogNode.assignWholeStr()

                # initialize the trees
                everyDogTree = getMono.CCGtree(NonTermNode=everyDogNode)
                someDogTree = getMono.CCGtree(NonTermNode=someDogNode)
                subjTree = getMono.CCGtree(TermNode=subj)

                # add pairs:
                # every dog < subj < some dog
                self.add_pair((everyDogTree, subjTree))
                self.add_pair((everyDogTree, someDogTree))
                self.add_pair((subjTree, someDogTree))
        print('...done!\n')
    """

    def quantifier_node(self, word):
        node_lemma = word
        return getMono.LeafNode(depth=0, cat=getMono.Cat('NP/N', word=node_lemma),
                                chunk=None, entity=None,
                                lemma=node_lemma, pos='DT', span=None, start=None,
                                word=node_lemma)

    def cardinal_node(self, word):
        node_lemma = word
        return getMono.LeafNode(depth=0, cat=getMono.Cat('NP/N', word=node_lemma),
                                chunk=None, entity=None,
                                lemma=node_lemma, pos='CD', span=None, start=None,
                                word=node_lemma)

    def build_quantifier(self, all_quant=True):
        r""" self-defined < relations for quantifiers: every < some, etc.
        all_quant = True: buld all quantifiers
        False: don't build a = an = some = one

        We need to do this separately for 
        - FraCas 
        - SICK, because "a = the" in SICK
        - data augmentation, because we don't want to generate a lot 
            of data with quantifiers (data becomes skewed)
        """
        ###################
        # FraCas
        node_every = self.quantifier_node('every')
        node_all = self.quantifier_node('all')
        node_each = self.quantifier_node('each')
        node_some = self.quantifier_node('some')
        node_a = self.quantifier_node('a')
        node_an = self.quantifier_node('an')
        node_one = self.quantifier_node('one')
        node_many = self.quantifier_node('many')
        node_several = self.quantifier_node('several')
        node_few = self.quantifier_node('few')
        node_most = self.quantifier_node('most')
        node_the = self.quantifier_node('the')
        node_at_least_N = self.quantifier_node('at-least-N')

        # every = all = each < most < many < a few = several < at least 3
        # < at least 2 = some2 < some1 = at least 1 = a
        # most of the time, we have some2. TODO for now just treat some as some2
        list1 = [node_every, node_all, node_each]
        list2 = [node_most]
        list3 = [node_many]
        list4 = [node_several]
        if all_quant:
            list5 = [node_some, node_a, node_an, node_one]
        else:
            list5 = [node_some, node_a]

        # small < big
        for small in list1: 
            for big in list2 + list3 + list4 + list5:
            # for big in [node_most, node_many, node_several, node_some, node_a, node_an]:
                self.add_pair((small, big))
        for big in [node_many, node_several, node_some, node_a, node_an]:
            self.add_pair((node_most, big))
        
        for big in [node_several, node_some, node_a, node_an]:
            self.add_pair((node_many, big))
        
        for big in [node_some, node_a, node_an]:
            self.add_pair((node_several, big))

        # equal
        try:
            self.add_equal_pair(list1[0], list1[1])
            self.add_equal_pair(list1[0], list1[2])
            self.add_equal_pair(list1[1], list1[2])
            self.add_equal_pair(list5[0], list5[1])
            self.add_equal_pair(list5[0], list5[2])
            self.add_equal_pair(list5[0], list5[3])
            self.add_equal_pair(list5[1], list5[2])
            self.add_equal_pair(list5[1], list5[3])
            self.add_equal_pair(list5[2], list5[3])
        except IndexError:
            pass

        # the < some1 = a = an = one
        self.add_pair((node_the, node_some))
        self.add_pair((node_the, node_a))
        if all_quant:
            self.add_pair((node_the, node_an))
            self.add_pair((node_the, node_one))


        # TODO few = at most N, similar to no, fixed in mark_leafnodes
        # few (students)\down (like books)\down
        # what's between ``few'' and some/every/no ??? TODO

        # at least N < some = a = an
        if all_quant:
            self.add_pair((node_at_least_N, node_some))
            self.add_pair((node_at_least_N, node_a))
            self.add_pair((node_at_least_N, node_an))

        # for Fracas
        self.add_pair((node_at_least_3, node_some))
        self.add_pair((node_several, node_at_least_several)) # id 27

        # --------------------------------------
        # cardinals
        node_2 = self.cardinal_node('2')
        node_3 = self.cardinal_node('3')
        node_4 = self.cardinal_node('4')
        node_5 = self.cardinal_node('5')
        self.add_pair((node_5, node_4))
        self.add_pair((node_5, node_3))
        self.add_pair((node_5, node_2))
        self.add_pair((node_4, node_3))
        self.add_pair((node_4, node_2))
        self.add_pair((node_3, node_2))

        # several
        # 5 < 4 < 3 < several
        self.add_pair((node_3, node_several))
        self.add_pair((node_4, node_several))
        self.add_pair((node_5, node_several))

        for small in [node_2, node_3, node_4, node_5]:
            for big in [node_some, node_a, node_an, node_one]:
                self.add_pair((small, big))
        

        return
        ###################
        # SICK
        # TODO for SICK: let's have: the = some = a = an = one CAUTION!!
        # self.add_equal_pair(node_the, node_some)
        # self.add_equal_pair(node_the, node_a)
        # self.add_equal_pair(node_the, node_an)
        # self.add_equal_pair(node_the, node_one)


        ###################
        # data augmentation


    def build_morph_tense(self):
        """ add self-defined morphology and tense to k
        e.g. men = man, want = wants = wanted 
        all this should be handled by working on LEMMAs instead of tokens,
        but because the tagger/parser makes mistakes, we have to manually
        add some pre-orders        
        """
        node_win = getMono.LeafNode(depth=0, cat=getMono.Cat(r'(S\NP)/NP', word='win'),
                                    chunk=None, entity=None,
                                    lemma='win', pos='VBP', span=None, start=None,
                                    word='win')
        node_won = getMono.LeafNode(depth=0, cat=getMono.Cat(r'(S\NP)/NP', word='won'),
                                    chunk=None, entity=None,
                                    lemma='won', pos='VBZ', span=None, start=None,
                                    word='won')
        # tree_win = getMono.CCGtree(TermNode=node_win)
        # tree_won = getMono.CCGtree(TermNode=node_won)

        self.add_equal_pair(node_win, node_won)

        node_europeans = getMono.LeafNode(depth=0, cat=getMono.Cat('N', word='Europeans'),
                                          chunk=None, entity=None,
                                          lemma='Europeans', pos='NNS', span=None, start=None,
                                          word='Europeans')
        node_european = getMono.LeafNode(depth=0, cat=getMono.Cat('N', word='European'),
                                         chunk=None, entity=None,
                                         lemma='European', pos='NN', span=None, start=None,
                                         word='European')
        # tree_europeans = getMono.CCGtree(TermNode=node_europeans)
        # tree_european = getMono.CCGtree(TermNode=node_european)
        self.add_equal_pair(node_european, node_europeans)

        r"""
        # nouns
        node_men = getMono.LeafNode(depth=0, cat=getMono.Cat('N', word='men'),
                                    chunk=None, entity=None,
                                    lemma='man', pos='NNS', span=None, start=None,
                                    word='men')
        node_man = getMono.LeafNode(depth=0, cat=getMono.Cat('N', word='man'),
                                    chunk=None, entity=None,
                                    lemma='man', pos='NN', span=None, start=None,
                                    word='man')
        node_tenors = getMono.LeafNode(depth=0, cat=getMono.Cat('N', word='tenors'),
                                       chunk=None, entity=None,
                                       lemma='tenor', pos='NNS', span=None, start=None,
                                       word='tenors')
        node_tenor = getMono.LeafNode(depth=0, cat=getMono.Cat('N', word='tenor'),
                                      chunk=None, entity=None,
                                      lemma='tenor', pos='NN', span=None, start=None,
                                      word='tenor')
        node_europeans = getMono.LeafNode(depth=0, cat=getMono.Cat('N', word='Europeans'),
                                       chunk=None, entity=None,
                                       lemma='European', pos='NNS', span=None, start=None,
                                       word='Europeans')
        node_european = getMono.LeafNode(depth=0, cat=getMono.Cat('N', word='European'),
                                      chunk=None, entity=None,
                                      lemma='European', pos='NN', span=None, start=None,
                                      word='European')

        # verbs
        node_want = getMono.LeafNode(depth=0, cat=getMono.Cat(r'(S\NP)/(S\NP)', word='want'),
                                     chunk=None, entity=None,
                                     lemma='want', pos='VBP', span=None, start=None,
                                     word='want')
        node_wants = getMono.LeafNode(depth=0, cat=getMono.Cat(r'(S\NP)/(S\NP)', word='wants'),
                                      chunk=None, entity=None,
                                      lemma='want', pos='VBZ', span=None, start=None,
                                      word='wants')
        node_wanted = getMono.LeafNode(depth=0, cat=getMono.Cat(r'(S\NP)/(S\NP)', word='wanted'),
                                       chunk=None, entity=None,
                                       lemma='want', pos='VBD', span=None, start=None,
                                       word='wanted')
        node_have = getMono.LeafNode(depth=0, cat=getMono.Cat(r'(S\NP)/NP', word='have'),
                                     chunk=None, entity=None,
                                     lemma='have', pos='VBP', span=None, start=None,
                                     word='have')
        node_has = getMono.LeafNode(depth=0, cat=getMono.Cat(r'(S\NP)/NP', word='has'),
                                    chunk=None, entity=None,
                                    lemma='have', pos='VBZ', span=None, start=None,
                                    word='has')

        # nouns
        tree_men = getMono.CCGtree(TermNode=node_men)
        tree_man = getMono.CCGtree(TermNode=node_man)
        tree_tenors = getMono.CCGtree(TermNode=node_tenors)
        tree_tenor = getMono.CCGtree(TermNode=node_tenor)
        tree_europeans = getMono.CCGtree(TermNode=node_europeans)
        tree_european = getMono.CCGtree(TermNode=node_european)

        # verbs
        tree_want = getMono.CCGtree(TermNode=node_want)
        tree_wants = getMono.CCGtree(TermNode=node_wants)
        tree_wanted = getMono.CCGtree(TermNode=node_wanted)
        tree_have = getMono.CCGtree(TermNode=node_have)
        tree_has = getMono.CCGtree(TermNode=node_has)

        # men = man, tenors = tenor
        self.add_equal_pair(tree_man, tree_men)
        self.add_equal_pair(tree_tenor, tree_tenors)
        self.add_equal_pair(tree_european, tree_europeans)

        # want = wants = wanted
        self.add_equal_pair(tree_want, tree_wants)
        self.add_equal_pair(tree_want, tree_wanted)
        self.add_equal_pair(tree_wants, tree_wanted)
        self.add_equal_pair(tree_has, tree_have)
        """

    def build_manual_for_sick(self):
        """ manually add x < y to KB """
        # return
        # ----------------------------------------
        # NOUNS
        # tree < plant
        self.add_pair_2_words(cat1='N', word1='tree', lemma1='tree', pos1='NN',
                              cat2='N', word2='plant', lemma2='plant', pos2='NN',
                              relation='<')

        # lady = woman
        self.add_pair_2_words(cat1='N', word1='lady', lemma1='lady', pos1='NN',
                              cat2='N', word2='woman', lemma2='woman', pos2='NN',
                              relation='=')

        # lady = girl
        self.add_pair_2_words(cat1='N', word1='lady', lemma1='lady', pos1='NN',
                              cat2='N', word2='girl', lemma2='girl', pos2='NN',
                              relation='=')

        # lady < person
        self.add_pair_2_words(cat1='N', word1='lady', lemma1='lady', pos1='NN',
                              cat2='N', word2='person', lemma2='person', pos2='NN',
                              relation='<')

        # note < paper
        self.add_pair_2_words(cat1=r'N', word1='note', lemma1='note', pos1='NN',
                              cat2=r'N', word2='paper', lemma2='paper', pos2='NN',
                              relation='<')

        # ringer < wrestler
        self.add_pair_2_words(cat1=r'N', word1='ringer', lemma1='ringer', pos1='NN',
                              cat2=r'N', word2='wrestler', lemma2='wrestler', pos2='NN',
                              relation='<')

        # food = meal
        # self.add_pair_2_words(cat1=r'N', word1='food', lemma1='food', pos1='NN',
        #                       cat2=r'N', word2='meal', lemma2='meal', pos2='NN',
        #                       relation='=')

        # logs = wood
        self.add_pair_2_words(cat1=r'N', word1='logs', lemma1='log', pos1='NNS',
                              cat2=r'N', word2='wood', lemma2='wood', pos2='NN',
                              relation='=')

        # drawing < tattoo
        self.add_pair_2_words(cat1=r'N/PP', word1='drawing', lemma1='drawing', pos1='NN',
                              cat2=r'N/PP', word2='tattoo', lemma2='tattoo', pos2='NN',
                              relation='=')

        # street = road
        self.add_pair_2_words(cat1=r'N', word1='street', lemma1='street', pos1='NN',
                              cat2=r'N', word2='road', lemma2='road', pos2='NN',
                              relation='=')

        # bikini < swimming suite
        self.add_pair_2_words(cat1=r'N', word1='bikini', lemma1='bikini', pos1='NN',
                              cat2=r'N', word2='swimming suite', lemma2='swimming suite', pos2='NN',
                              relation='<')

        # people = person
        self.add_pair_2_words(cat1=r'N', word1='people', lemma1='people', pos1='NNS',
                              cat2=r'N', word2='person', lemma2='person', pos2='NN',
                              relation='=')

        # schoolgirl < girl
        self.add_pair_2_words(cat1=r'N/PP', word1='schoolgirl', lemma1='schoolgirl', pos1='NN',
                              cat2=r'N/PP', word2='girl', lemma2='girl', pos2='NN',
                              relation='<')

        # stunt < trick
        self.add_pair_2_words(cat1=r'N', word1='stunt', lemma1='stunt', pos1='NN',
                              cat2=r'N', word2='trick', lemma2='trick', pos2='NN',
                              relation='<')

        # act < perform


        # ----------------------------------------
        # VERBS:
        # strum < play
        self.add_pair_2_words(cat1=r'(S\NP)/NP', word1='strum', lemma1='strum', pos1='VBZ',
                              cat2=r'(S\NP)/NP', word2='play', lemma2='play', pos2='VBZ',
                              relation='<')

        # hurl = throw?
        self.add_pair_2_words(cat1=r'(S\NP)/NP', word1='hurl', lemma1='hurl', pos1='VBZ',
                              cat2=r'(S\NP)/NP', word2='throw', lemma2='throw', pos2='VBZ',
                              relation='=')

        # bounce < jump
        self.add_pair_2_words(cat1=r'S\NP', word1='bounce', lemma1='bounce', pos1='VBZ',
                              cat2=r'S\NP', word2='jump', lemma2='jump', pos2='VBZ',
                              relation='<')

        # lunge < jump
        self.add_pair_2_words(cat1=r'(S\NP)/PP', word1='lunge', lemma1='lunge', pos1='VBZ',
                              cat2=r'(S\NP)/PP', word2='jump', lemma2='jump', pos2='VBZ',
                              relation='<')

        # slice < cut
        self.add_pair_2_words(cat1=r'(S\NP)/NP', word1='slice', lemma1='slice', pos1='VBZ',
                              cat2=r'(S\NP)/NP', word2='cut', lemma2='cut', pos2='VBZ',
                              relation='<')

        # speak = talk
        self.add_pair_2_words(cat1=r'(S\NP)/PP', word1='speak', lemma1='speak', pos1='VBZ',
                              cat2=r'(S\NP)/PP', word2='talk', lemma2='talk', pos2='VBZ',
                              relation='=')

        # saw < cut
        self.add_pair_2_words(cat1=r'(S\NP)/NP', word1='saw', lemma1='saw', pos1='VBZ',
                              cat2=r'(S\NP)/NP', word2='cut', lemma2='cut', pos2='VBZ',
                              relation='<')
        # pause < stop
        self.add_pair_2_words(cat1=r'S\NP', word1='pause', lemma1='pause', pos1='VBZ',
                              cat2=r'S\NP', word2='stop', lemma2='stop', pos2='VBZ',
                              relation='<')

        self.add_pair_2_words(cat1=r'(S\NP)/NP', word1='miss', lemma1='miss', pos1='VBZ',
                              cat2=r'(S\NP)/NP', word2='catch', lemma2='catch', pos2='VBZ',
                              relation='<>')

        # hike < walk
        self.add_pair_2_words(cat1=r'S\NP', word1='hike', lemma1='hike', pos1='VBZ',
                              cat2=r'S\NP', word2='walk', lemma2='walk', pos2='VBZ',
                              relation='<')

        # ----------------------------------------
        # ADJ
        # large = big
        self.add_pair_2_words(cat1=r'N/N', word1='large', lemma1='large', pos1='JJ',
                              cat2=r'N/N', word2='big', lemma2='big', pos2='JJ',
                              relation='=')
        # huge = big
        self.add_pair_2_words(cat1=r'N/N', word1='huge', lemma1='huge', pos1='JJ',
                              cat2=r'N/N', word2='big', lemma2='big', pos2='JJ',
                              relation='=')

        self.add_pair_2_words(cat1=r'N/N', word1='small', lemma1='small', pos1='JJ',
                              cat2=r'N/N', word2='big', lemma2='big', pos2='JJ',
                              relation='<>')

        # ----------------------------------------
        # prepositions
        # on <> off
        self.add_pair_2_words(cat1=r'PR', word1='on', lemma1='on', pos1='RP',
                              cat2=r'PR', word2='off', lemma2='off', pos2='RP',
                              relation='<>')
        self.add_pair_2_words(cat1=r'((S\NP)\(S\NP))/NP', word1='on', lemma1='on', pos1='IN',
                              cat2=r'((S\NP)\(S\NP))/NP', word2='off', lemma2='off', pos2='IN',
                              relation='<>')
        # up <> down
        # self.add_pair_2_words(cat1=r'PR', word1='up', lemma1='up', pos1='RP',
        #                       cat2=r'PR', word2='down', lemma2='down', pos2='RP',
        #                       relation='<>')
        self.add_pair_2_words(cat1=r'((S\NP)\(S\NP))/NP', word1='up', lemma1='up', pos1='IN',
                              cat2=r'((S\NP)\(S\NP))/NP', word2='down', lemma2='down', pos2='IN',
                              relation='<>')
        # inside <> outside
        self.add_pair_2_words(cat1=r'(S\NP)\(S\NP)', word1='outside', lemma1='outside', pos1='IN',
                              cat2=r'(S\NP)\(S\NP)', word2='inside', lemma2='inside', pos2='IN',
                              relation='<>')

    def add_pair_2_words(self, cat1, word1, lemma1, pos1, cat2, word2, lemma2, pos2, relation, number1=None, number2=None):
        """ add a pair relation to KB, x < y, or x = y, or x <> y """
        node_1 = utils.make_leafnode(cat=cat1, word=word1, lemma=lemma1, pos=pos1, number=number1)
        node_2 = utils.make_leafnode(cat=cat2, word=word2, lemma=lemma2, pos=pos2, number=number2)
        if relation == "<":
            self.add_pair((node_1, node_2))
        elif relation == "=":
            self.add_equal_pair(node_1, node_2)
        elif relation == "<>":
            self.add_antonym_pair(node_1, node_2)

    # def conjunction(self, tree):
    #     """
    #     !! only needed for generating data for semantics fragment paper at AAAI 2020 !!
    #     add to k: A and B < A/B < A or B, where A, B are NPs
    #     """
    #     for node in tree.nonTermNodes:
    #         if node.cat.typeWOfeats == "NP":
    #             # eprint()
    #             # eprint(node)
    #             # eprint(node.wholeStr)
    #             # eprint()
    #             if "AND" in node.wholeStr:
    #                 # eprint("option 1\n")
    #                 # A and B < A/B < A or B, where A
    #                 #
    #                 #      and   NP2
    #                 #     ----------
    #                 #  NP1    NP\NP
    #                 # ------------
    #                 #         NP = node
    #                 try: assert node.children[1].children[0].wholeStr == "AND"
    #                 except AssertionError: continue
    #                 NP1 = node.children[0]
    #                 NP2 = node.children[1].children[1]
    #                 self.add_pair((node, NP1), True)
    #                 self.add_pair((node, NP2), True)
    #                 # get "A or B"
    #                 node_NP1_or_NP2 = deepcopy(node)
    #                 # change "and" to "or"
    #                 node_NP1_or_NP2.children[1].children[0].assign_new_word(word="or", number=None)
    #                 self.add_pair((node, node_NP1_or_NP2), True)
    #                 self.add_pair((NP1, node_NP1_or_NP2), True)
    #                 self.add_pair((NP2, node_NP1_or_NP2), True)
    #             elif node.children[0].cat.typeWOfeats == "NP/N":
    #                 # eprint("option 2\n")
    #                 # NP and NP_new < NP/NP_new
    #                 # TODO
    #                 NP = node
    #                 NP_new = NP_ani()
    #                 head_noun_NP = find_head_noun(node_det=NP.children[0], ccg_tree=NP)
    #                 head_noun_NP_new = find_head_noun(NP_new.children[0], NP_new)
    #                 while same_ani_noun(head_noun_NP, head_noun_NP_new):
    #                     NP_new = NP_ani()
    #                     head_noun_NP_new = find_head_noun(NP_new.children[0], NP_new)
    #                 # get the conjoined NP
    #                 #             and        no cat
    #                 #             (NP\NP)/NP    NP_new
    #                 #  some dog   ---------------- conj
    #                 #   NP1 = NP        NP\NP = node_1
    #                 # --------------------------
    #                 #          NP (Plural) = node_new
    #                 node_and = node_and_for_NP(cat=r"(NP\NP)/NP")
    #                 node_1 = make_nontermnode(cat=r"NP\NP", ruleType="conj", number=NP_new.number)
    #                 set_parent_children(node_1, [node_and, NP_new])
    #                 node_new = make_nontermnode(cat=r"NP", ruleType="ba", number='pl')
    #                 set_parent_children(node_new, [NP, node_1])
    #                 # add to k
    #                 self.add_pair((node_new, NP), True)
    #                 self.add_pair((node_new, NP_new), True)
    #                 # TODO: to test

class Fragment:
    def __init__(self, ccgtree=None):
        '''  small < frag < big;  a frag could be "chased some cat"  '''
        self.ccgtree = ccgtree  # a CCGtree
        self.wholeStr = ccgtree.wholeStr  # 'CHASED SOME CAT'
        self.small = []  # could have multiple small
        self.big = []    # could have multiple big
        self.equal = []  # could have multiple equal
        self.ant = []    # antonyms
        self.alter = []  # alternatives: dog | cat TODO

    def __str__(self):
        return '{}'.format(self.wholeStr)

    def __repr__(self):
        return self.__str__()

'''
!!! OLD !!!
Pipeline:

wordnet/sentences => our representation => replacement

Now we just assume we already have our representation.
In the next step, we start with wordnet/sentences

===================
Our representation (Knowledge class):

- big dogs < large animals

Both phrases should be trees. I.e. the CCGtree class from getMono.py

big   dogs
N/N   N
---------- fa
    N

===================
Replacement:

- Find in the premise a node N 1) with string 'big dogs' and
2) with the same tree structure (?this will be hard to do)

- Replace N with the node 'large animals'

We should loop every node in CCGtree.

===================
Test:
- premise: Every large animal likes vegetables.
- conc: Every big dog likes carrots.

'''


def build_knowledge_test():
    print('\n\n-----------\nbuilding knowledge...')
    k = Knowledge()

    '''
    I chased some cat.
    I hit every bird.
    I see some animal.
    It is black.

    chased some cat < liked every dog (small1, big1)
    see some animal < is black (small2, big2)
    old dog < dog (small3, big3)
    young man < man (s4, b4)
    cat < animal (s5, b5)
    every man < John < some man (s6, m6, b6)
    '''

    knowledge_trees = getMono.CCGtrees()
    knowledge_trees.readCandCxml('knowledge.candc.xml')

    # ----------------------
    small1 = getMono.CCGtree(
        NonTermNode=knowledge_trees.trees[0].root.children[0].children[1]
    )
    print(small1.wholeStr)

    big1 = getMono.CCGtree(
        NonTermNode=knowledge_trees.trees[1].root.children[0].children[1]
    )
    print(big1.wholeStr)

    # ----------------------
    small2 = getMono.CCGtree(
        NonTermNode=knowledge_trees.trees[2].root.children[0].children[1]
    )
    print(small2.wholeStr)

    big2 = getMono.CCGtree(
        NonTermNode=knowledge_trees.trees[3].root.children[0].children[1]
    )
    print(big2.wholeStr)

    # ----------------------
    small3 = getMono.CCGtree(
        NonTermNode=knowledge_trees.trees[4].root.children[0].children[1]
    )
    print(small3.wholeStr)

    big3 = getMono.CCGtree(
        NonTermNode=knowledge_trees.trees[5].root.children[0].children[1]
    )
    print(big3.wholeStr)

    # ----------------------
    s4 = getMono.CCGtree(
        NonTermNode=knowledge_trees.trees[0].root.children[0].children[1].children[1].children[1]
    )
    print(s4.wholeStr)  # cat

    b4 = getMono.CCGtree(
        NonTermNode=knowledge_trees.trees[6].root.children[0].children[1]
    )
    print(b4.wholeStr)  # animal

    # ----------------------
    s5 = getMono.CCGtree(
        NonTermNode=knowledge_trees.trees[7].root.children[0].children[1]
    )
    print(s5.wholeStr)  # young man

    b5 = getMono.CCGtree(
        NonTermNode=knowledge_trees.trees[7].root.children[0].children[1].children[1]
    )
    print(b5.wholeStr)  # man

    # ----------------------
    s6 = getMono.CCGtree(
        NonTermNode=knowledge_trees.trees[8].root.children[0].children[0]
    )
    print(s6.wholeStr)  # every man

    m6 = getMono.CCGtree(
        NonTermNode=knowledge_trees.trees[9].root.children[0].children[0]
    )
    print(m6.wholeStr)  # John

    b6 = getMono.CCGtree(
        NonTermNode=knowledge_trees.trees[10].root.children[0].children[0]
    )
    print(b6.wholeStr)  # some man

    # add to knowledge
    k.add_pair((small1, big1))
    k.add_pair((small2, big2))
    k.add_pair((small3, big3))
    k.add_pair((s4, b4))
    k.add_pair((s5, b5))
    k.add_pair((s6, m6))
    k.add_pair((m6, b6))
    k.add_pair((s6, b6))

    print('\nknowledge built!\n--------------\n')
    return k


if __name__ == '__main__':
    main()

