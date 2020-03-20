#!/usr/bin/env python3
"""
Logical systems

if input = John is a linguist,
*KB*: add ``every linguist < John < some linguist''

if input = Every European has the right to live in Europe
*KB*: add ``every European < has the right to live in Europe''
*LogicalSys*: add ``(every), (European, has the right to live in Europe)''

if input = Most Europeans are resident in Europe
*KB*: add nothing
*LogicalSys*: add ``(most), (Europeans, resident in Europe)''

if input = I ate an apple today,
add nothing to both KB and LogicalSys

if input = Several delegates got the results published in major national newspapers.
*KB*: add nothing
*LogicalSys*: add ``(several), (delegates, got the ... newspapers)''

if input = All people who are resident in Europe can travel freely within Europe.
*KB*: add ``people who are resident in Europe < can travel freely within Europen''
*LogicalSys*: add ``(all), (people who ..., can travel ...)''

if input = Some Italian tenors are great
*LogicalSys*: add ``(some), (Italian tenors, great)'' ??? great is ADJ

#####################################

syntax of LogicalSys:

    quantifier X Y

where X = N/NP, Y = N/NP/VP/ADJ

what about relative clauses? Actually, relative clauses only becomes
a problem when we translate from LogicalSys to natural sentences

#####################################

rules:

- DET rule:

DET  x  y        All  x  z
-------------------------  DET: several cases based on pos of y and z
      DET x (y ^ z)
Note: z is a verb

- CONSERVATIVITY rule:

DET x y
--------------CONSERV: y can be N/NP/VP/ADJ
DET x (x ^ y)

- SOME:

SOME x y
---------  SOME_1: always works, b/c x is always N/NP
SOME x x

SOME x y
---------  SOME_2: only works when x, y are both N/NP
SOME y x


"""

__author__ = "Hai Hu"

import sys, argparse, os, time
from getMono import CCGtrees, CCGtree, LeafNode, NonTermNode, Cat,\
    ErrorCCGtree, ErrorCompareSemCat, eprint
from copy import deepcopy
import utils

QUANTIFIERS = {'EVERY', 'ALL', 'EACH', 'MOST', 'MANY', 'SEVERAL', 'SOME', 'A', 'AN',
               'NO', 'FEW', 'BOTH', 'NEITHER', 'EITHER'}

# quantifiers greater than or equal to
QUANT_GEQ = {
    'EVERY' : {'EVERY', 'ALL', 'EACH', 'MOST', 'MANY', 'SEVERAL', 'SOME', 'A', 'AN'},
    'ALL': {'EVERY', 'ALL', 'EACH', 'MOST', 'MANY', 'SEVERAL', 'SOME', 'A', 'AN'},
    'EACH' : {'EVERY', 'ALL', 'EACH', 'MOST', 'MANY', 'SEVERAL', 'SOME', 'A', 'AN'},
    'MOST': {'MOST', 'MANY', 'SEVERAL', 'SOME', 'A', 'AN'},
    'MANY' : {'MANY', 'SEVERAL', 'SOME', 'A', 'AN'},
    'SEVERAL': {'SEVERAL', 'SOME', 'A', 'AN'},
    'SOME': {'SOME', 'A', 'AN'},
    'A': {'SOME', 'A', 'AN'},
    'AN': {'SOME', 'A', 'AN'}
}

def main():
    test()
    # test1()


def test():
    trees = CCGtrees("fracas_1_80.raw.tok.preprocess.log")
    trees.readEasyccgStr("fracas_1_80.raw.easyccg.parsed.txt")
    from infer import Knowledge
    k = Knowledge()
    k.build_quantifier()         # all = every = each < some = a = an, etc.
    k.build_morph_tense()         # man = men, etc.

    LS = LogicalSystem(k)
    LS.manual()

    for i in range(len(trees.trees)):
        if i not in [40, 41, 42]: continue
        # if i not in [64, 65, 66]: continue
        # if i not in [44, 45, 46]: continue
        tree = trees.trees[i]
        tree.fixQuantifier()
        tree.fixNot()
        LS.add_logical_sent(tree)

        k.update_sent_pattern(tree)  # patterns like: every X is NP
        k.update_word_lists(tree)    # nouns, subSecAdj, etc.

    k.print_knowledge()

    LS.compute_DET_rule()
    print(LS)

def test1():
    from infer import Knowledge, Fragment
    k = Knowledge()
    k.build_quantifier()  # all = every = each < some = a = an, etc.
    k.build_morph_tense()  # man = men, etc.

    # question: if we have DET x' y, we should also add DET x (y^z), DET x' (y^z)
    # DET  x  y        All  x  z
    # -------------------------  DET: several cases based on pos of y and z
    #        DET x (y ^ z)

    LS = LogicalSystem(k)
    logsent1 = LogicalSentence("every", utils.node_X, utils.node_Z)
    logsent2 = LogicalSentence("most", utils.node_X_bar, utils.node_Y)
    LS.add_logical_sent_helper(logsent1)
    LS.add_logical_sent_helper(logsent2)

    # add equal relations between X and X bar
    k.add_equal_pair(utils.node_X, utils.node_X_bar)

    # k.print_knowledge()

    # chekc if DET x (y^z), DET x' (y^z) is added
    for inf in LS.compute_DET_rule():
        print(inf)

    print(LS)

class LogicalSystem:
    def __init__(self, KB):
        """  KB: KnowledgeBase
        'log_sents' : set of log_sents
        """
        self.logical_sents = {
            q : { 'log_sents' : set(), 'strs' : set(),
                  'X' : set(), 'Y' : set() } \
            for q in QUANTIFIERS
        }
        self.KB = KB

        self.counter_sent_start_w_quant = 0
        self.counter_add_logical_sent = 0

    def add_logical_sent(self, sent_ccgtree):
        """ read in a ccgtree, and translate it to logical sent, add to logical_sents """
        # test case fracas-026: most Europeans are resident in Europe
        # we want: self.quant_rel = {'MOST': set( (node_A, node_B), (), ... ) }
        # node_A = Europeans, node_B = resident in Europe
        # TODO the parse from easyccg is wrong!
        if sent_ccgtree.wholeStr.split(' ')[0] in QUANTIFIERS:
            self.counter_sent_start_w_quant += 1
            # TODO case 1: VP = be + X
            if "BE" in sent_ccgtree.words:
                self.quantifier_x_be_y(sent_ccgtree)

            # TODO case 2: VP = regular VP
            else:
                self.quantifier_x_VP(sent_ccgtree)

    def quantifier_x_be_y(self, sent_ccgtree):
        """  add ``quant (N, NP_2)'' to logical_sents """
        # TODO how to know if this "BE" is the main verb
        # "BE" can be at multiple places
        # match structure:
        # most    A           are        B
        # NP/N    N          (S\NP)/NP   NP_2 = X
        # --------------     ----------------
        #      NP_1                S\NP  =  VP
        #
        node_most = sent_ccgtree.getLeftMostLeaf(sent_ccgtree.root)
        try: assert node_most.wholeStr in QUANTIFIERS
        except AssertionError:
            print("something wrong adding sent to logicalSys:")
            sent_ccgtree.printSent()
            exit()

        quant = node_most.wholeStr
        N, X, VP = None, None, None
        if node_most.sisters: N = node_most.sisters[0]
        if node_most.parent.sisters: VP = node_most.parent.sisters[0]
        if VP and (VP.cat.typeWOfeats == r"S\NP") and \
                VP.children and (VP.children[0].wholeStr == "BE"):
            # case a: X = plural nouns
            # case b: X = a person? we only need ``person''?
            # case c: Every person is great. be: (S\NP)/(S\NP), great: S\NP
            # case d: Every one is good at dancing. same as above
            # case e: Every one is about to dance. same as above
            # case f: Several committee members are from Scandinavia. PP

            # case b
            if len(VP.children[1].children) == 2 and \
                            VP.children[1].children[0].wholeStr in ["A", "AN"]:
                X = VP.children[1].children[1]

            # case a,c,d,e,f
            elif VP.children[1].cat.typeWOfeats in {"NP", r"S\NP", "PP"}:
                X = VP.children[1]

            else:
                X = VP.children[1]

            if N and X:
                log_sent = LogicalSentence(quant, N, X)
                self.add_logical_sent_helper(log_sent)
            else:
                eprint('\ndid not add to logical sys 1:')
                sent_ccgtree.printSent()

        else:  # "BE" is not the main verb!
            self.quantifier_x_VP(sent_ccgtree)

    def quantifier_x_VP(self, sent_ccgtree):
        """  add ``quant (N, VP)'' to logical_sents """
        # match structure:
        # most    A
        # NP/N    N               regular VP
        # --------------     ----------------
        #      NP_1                S\NP  =  VP
        #
        node_most = sent_ccgtree.getLeftMostLeaf(sent_ccgtree.root)
        try: assert node_most.wholeStr in QUANTIFIERS
        except AssertionError:
            print("something wrong adding sent to logicalSys:")
            sent_ccgtree.printSent()
            exit()

        quant = node_most.wholeStr
        N, VP = None, None
        if node_most.sisters: N = node_most.sisters[0]
        if node_most.parent.sisters: VP = node_most.parent.sisters[0]
        if VP and VP.cat.typeWOfeats == r"S\NP":
            if N:
                log_sent = LogicalSentence(quant, N, VP)
                self.add_logical_sent_helper(log_sent)
            else:
                eprint('\ndid not add to logical sys 2.1:')
                sent_ccgtree.printSent()
        else:
            eprint('\ndid not add to logical sys 2.2:')
            sent_ccgtree.printSent()

    def add_logical_sent_helper(self, log_sent):
        """ add `quant node1 node2' to self.logical_sents, e.g. `most dogs animals' """
        # node1 = log_sent.term1
        # node1_equals = [node1]
        # if node1.wholeStr in self.KB.frags:
        #     node1_equals.extend([i.ccgtree.root for i in self.KB.frags[node1.wholeStr].equal])
        self.counter_add_logical_sent += 1
        quant = log_sent.quantifier.upper()
        if log_sent.str_terms not in self.logical_sents[quant]['strs']:
            self.logical_sents[quant]['log_sents'].add( log_sent )  # add LogicalSentence
            self.logical_sents[quant]['strs'].add( log_sent.str_terms )  # add string
            self.logical_sents[quant]['X'].add(log_sent.term1.wholeStr)  # add term1.str
            self.logical_sents[quant]['Y'].add(log_sent.term2.wholeStr)  # add term2.str

    def compute_DET_rule(self):
        """ apply DET_rule on all logical_sents
        return inferences to be added to the fringe!

        DET  x  y        All  x  z
        -------------------------  DET: several cases based on pos of y and z
              DET x (y ^ z)

        all possibilities of y ^ z are stored in a list, returned from conjoin_two_terms(y, z)
        """

        # step 1: find 'all/each/every x z'
        to_loop = [
            self.logical_sents['ALL']['log_sents'],
            self.logical_sents['EACH']['log_sents'],
            self.logical_sents['EVERY']['log_sents']
                   ]
        for lst in to_loop:
            for log_sent in lst:  # log_sent: All x z
                x = log_sent.term1  # NonTermNode
                z = log_sent.term2

                # x and its equivalents, a set of str
                x_set = set([i.ccgtree.root for i in self.KB.frags[x.wholeStr].equal] +
                                [x])
                x_set_str = set([i.wholeStr for i in x_set])
                # print('\n\n')
                # print(x_set_str)

                # step 2: find DET x y
                for quant in QUANTIFIERS:
                    # see if x and its equivalents appear as term1 in logSent for this quant
                    if not any([i in self.logical_sents[quant]['X'] for i in x_set_str ]):
                        continue  # then skip the following
                    for log_sent2 in self.logical_sents[quant]['log_sents']:
                        # log_sent2: DET x y
                        # if log_sent2.term1 is same as x, or the equavalents of x
                        if log_sent2.term1.wholeStr in x_set_str:
                            # found y!! DET rule here!!
                            # order is taken care of in the function
                            y = log_sent2.term2
                            y_and_z_list = self.conjoin_two_terms(y, z)  # a list of NonTermNode
                            for y_and_z in y_and_z_list:
                                print("\n\nquant", quant)
                                print("x", x.wholeStr)
                                print("y_and_z", y_and_z.wholeStr, y_and_z.cat.typeWOfeats)
                                # apart from adding ``quant x y_and_z'',
                                # we also need to add any quantifier greater or equal to quant
                                # e.g. every x y_and_z = all x y_and_z = each x y_and_z
                                if quant in QUANT_GEQ:
                                    new_quants_str = QUANT_GEQ[quant]
                                else: new_quants_str = [quant]

                                ### !!! KEY STEP !!! ### generate inf
                                for new_quant_str in new_quants_str:
                                    for new_x in x_set:
                                        inf_log_sent = LogicalSentence(quantifier=new_quant_str,
                                                                       term1=new_x, term2=y_and_z)
                                        inf_nat_sent = inf_log_sent.to_nat_sent()
                                        eprint("# DET_rule add to fringe: ", end="")
                                        inf_nat_sent.printSent(sys.stderr)
                                        yield inf_nat_sent

    def conjoin_two_terms(self, term1, term2):
        """  conjoin y and z in 'DET x (y ^ z)'
        return all possibilities of y ^ z as a list of NonTermNode """
        # y, z could be: 1) N, 2) NP, 3) S\NP, 4) PP
        # 3) S\NP can be: BRITISH(adj), GET THE RESULTS PUBLISHED(VP)
        # or possibly others

        type1, type2 = term1.cat.typeWOfeats, term2.cat.typeWOfeats
        ans = []  # a list of NonTermNodes

        # ans_node = NonTermNode(depth=0, cat=None, ruleType="fa", wholeStr="", impType="", note=None)

        # 10 cases
        # case 1 N + N: COMMITTEE MEMBER, RESIDENT OF THE NORTH AMERICAN CONTINENT
        if type1 == "N" and type2 == "N":
            # ans1: N1 who be N2
            ans.append(self.conjoin_two_terms_return_N(term1, term2))
            # ans2: N2 who be N1
            ans.append(self.conjoin_two_terms_return_N(term2, term1))

        # case 2 N + NP  TODO fracas needs this
        elif (type1 == "N" and type2 == "NP") or (type1 == "NP" and type2 == "N"):
            if type1 == "N": N = term1; NP = term2
            else: N = term2; NP = term1
            # ans1: N who be NP
            ans.append(self.conjoin_two_terms_return_N(N, NP))
            # ans2: NP who be N TODO
            ans.append(self.conjoin_two_terms_return_N(NP, N))
            pass

        # case 3 N + S\NP  TODO fracas needs this
        elif (type1 == "N" and type2 == r"S\NP") or (type1 == r"S\NP" and type2 == "N"):
            if type1 == "N": N = term1; tmp_P = term2
            else: N = term2; tmp_P = term1
            # tmp_P can be: adj or VP
            try:
                if tmp_P.pos == "JJ":  # leafnode, adj
                    adj = tmp_P
                    ans.append(self.conjoin_two_terms_return_N(N, adj, adj=True))

            except AttributeError:  # non term node, VP
                VP = tmp_P
                ans.append(self.conjoin_two_terms_return_N(N, VP))

        # case 4 N + PP

        # case 5 NP + NP

        # case 6 NP + S\NP

        # case 7 NP + PP  TODO fracas needs this
        elif (type1 == "NP" and type2 == "PP") or (type1 == "PP" and type2 == "NP"):
            if type1 == "NP": NP = term1; PP = term2
            else: NP = term2; PP = term1
            ans.append(self.conjoin_two_terms_return_N(NP, PP))

        # case 8 S\NP + S\NP

        # case 9 S\NP + PP

        # case 10 PP + PP

        return ans

    def conjoin_two_terms_return_N(self, term1, term2, adj=False):
        """  return one NonTermNode, which is (term1 ^ term2)  """
        term1 = deepcopy(term1)
        term2 = deepcopy(term2)

        if term1.cat.typeWOfeats in {"N", "NP"}:
            # return ``term1 who be term2''
            if term2.cat.typeWOfeats in {"N", "NP", "PP"} or adj:
                node_who = LeafNode(depth=0, cat=Cat(r'(N\N)/(S\NP)', word='who'),
                                    chunk=None, entity=None, lemma='who', pos='WP',
                                    span=None, start=None, word='who')
                node_be = LeafNode(depth=0, cat=Cat(r"(S\NP)/" + term2.cat.typeWOfeats, word='be'),
                                   chunk=None, entity=None, lemma='be', pos='VBP',
                                   span=None, start=None, word='be')

                node_be_term2 = NonTermNode(depth=0, cat=Cat(r"S\NP"), ruleType="fa")
                node_be_term2.set_children([node_be, term2])
                node_who_be_term2 = NonTermNode(depth=0, cat=Cat(r"N\N"), ruleType="fa")
                node_who_be_term2.set_children([node_who, node_be_term2])

                if term1.cat.typeWOfeats == "N":
                    node_term1_who_be_term2 = NonTermNode(depth=0, cat=Cat("N"), ruleType="ba")
                    node_term1_who_be_term2.set_children([term1, node_who_be_term2])
                else:  # term1 = "NP", first unlex it from NP to N
                    node_term1_N = NonTermNode(depth=0, cat=Cat("N"), ruleType="unlex")
                    node_term1_N.set_children([term1])
                    node_term1_who_be_term2 = NonTermNode(depth=0, cat=Cat("N"), ruleType="ba")
                    node_term1_who_be_term2.set_children([node_term1_N, node_who_be_term2])
                node_term1_who_be_term2.assignWholeStr()
                return node_term1_who_be_term2

            # return ``term1 who term2''
            elif term2.cat.typeWOfeats == r"S\NP":
                node_who = LeafNode(depth=0, cat=Cat(r'(N\N)/(S\NP)', word='who'),
                                    chunk=None, entity=None, lemma='who', pos='WP',
                                    span=None, start=None, word='who')
                node_who_term2 = NonTermNode(depth=0, cat=Cat(r"N\N"), ruleType="fa")
                node_who_term2.set_children([node_who, term2])

                if term1.cat.typeWOfeats == "N":
                    node_term1_who_term2 = NonTermNode(depth=0, cat=Cat("N"), ruleType="ba")
                    node_term1_who_term2.set_children([term1, node_who_term2])
                else:  # term1 = "NP", first unlex it from NP to N
                    node_term1_N = NonTermNode(depth=0, cat=Cat("N"), ruleType="unlex")
                    node_term1_N.set_children([term1])
                    node_term1_who_term2 = NonTermNode(depth=0, cat=Cat("N"), ruleType="ba")
                    node_term1_who_term2.set_children([node_term1_N, node_who_term2])
                node_term1_who_term2.assignWholeStr()
                return node_term1_who_term2

            else: pass
        else: pass

    def manual(self):
        """ manually add: most (Europeans, resident in Europe) """
        node_N_1 = LeafNode(
            depth=0, cat=Cat(r"N", word="Europeans"), chunk=None, entity=None, lemma="Europeans",
            pos="NNS", span=None, start=None, word="Europeans",
            impType=None, fixed=False)
        node_resident = LeafNode(
            depth=0, cat=Cat(r"N", word="resident"), chunk=None, entity=None, lemma="resident",
            pos="NN", span=None, start=None, word="resident",
            impType=None, fixed=False)
        node_in = LeafNode(
            depth=0, cat=Cat(r"(N\N)/NP", word="in"), chunk=None, entity=None, lemma="in",
            pos="IN", span=None, start=None, word="in",
            impType=None, fixed=False)
        node_europe = LeafNode(
            depth=0, cat=Cat("NP", word="Europe"), chunk=None, entity=None, lemma="Europe",
            pos="NNP", span=None, start=None, word="Europe",
            impType=None, fixed=False)
        node_in_europe = NonTermNode(depth=0, cat=Cat(r"N\N"), ruleType='fa')
        node_in_europe.set_children([node_in, node_europe])

        node_resident_in_europe = NonTermNode(depth=0, cat=Cat("N"), ruleType='ba')
        node_resident_in_europe.set_children([node_resident, node_in_europe])

        log_sent = LogicalSentence("MOST", node_N_1, node_resident_in_europe)
        self.add_logical_sent_helper(log_sent)

    def __str__(self):
        ans = "\nLogicalSystem:"
        for quant in sorted(self.logical_sents.keys()):
            ans += "\n\n---------------\n"
            ans += quant + ":\n\t"
            ans += "\n\t".join(sorted([log_sent.cat_terms + " " + log_sent.str_terms \
                                       for log_sent in self.logical_sents[quant]['log_sents']]))
        ans += "\n\n{} sentences starts with quantifiers".format(self.counter_sent_start_w_quant)
        ans += "\n\n{} LogicalSentence added".format(self.counter_add_logical_sent)
        return ans

    def __repr__(self):
        return self.__str__()

class LogicalSentence:
    def __init__(self, quantifier, term1, term2):
        """ e.g. all x y """
        self.quantifier = quantifier   # all, a string
        self.term1 = deepcopy(term1)   # x as a node
        self.term2 = deepcopy(term2)   # y as a node
        self.str_terms = term1.wholeStr + '  --  ' + term2.wholeStr  # "x_y"
        self.cat_terms = "{} -- {:10}".format(term1.cat.typeWOfeats, term2.cat.typeWOfeats)  # NP NP

    def to_nat_sent(self):
        """ translate a logical sentence to sentence in natural language
        return a CCGtree? or a sentence as a string? """
        # term1 must be N!
        # term2 = N or NP
        if self.term2.cat.typeWOfeats in {"N", "NP"}:
            # output: quantifier term1 be term2
            node_be = LeafNode(depth=0, cat=Cat(r"(S\NP)/" + self.term2.cat.typeWOfeats, word='be'),
                               chunk=None, entity=None, lemma='be', pos='VBP',
                               span=None, start=None, word='be')
            node_be_term2 = NonTermNode(depth=0, cat=Cat(r"S\NP"), ruleType="fa")
            node_be_term2.set_children([node_be, self.term2])
            node_quant = self.quant_2_node(quant=self.quantifier)
            node_quant_term1 = NonTermNode(depth=0, cat=Cat("NP"), ruleType="fa")
            node_quant_term1.set_children([node_quant, self.term1])
            node_s = NonTermNode(depth=0, cat=Cat("S"), ruleType="ba")
            node_s.set_children([node_quant_term1, node_be_term2])
            nat_sent = CCGtree(NonTermNode=node_s)

            nat_sent.set_markings_to_none() # first reset markings
            nat_sent.mark()
            nat_sent.polarize()
            return nat_sent
        else:
            pass

    def quant_2_node(self, quant):
        """ build a LeafNode for quant, e.g. all, every. quant = string """
        return LeafNode(depth=0, cat=Cat(r"NP/N"), chunk=0, entity=None, lemma=quant.lower(),
                        pos="DT", span=0, start=0, word=quant.lower())


class Term:
    def __init__(self):
        pass

if __name__ == '__main__':
    main()






