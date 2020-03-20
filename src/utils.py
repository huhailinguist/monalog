#!/usr/bin/env python3
"""
utils
pre-built LeafNodes

"""

__author__ = "Hai Hu"

from getMono import CCGtree, LeafNode, NonTermNode, Cat, eprint
import csv, getMono, difflib
import spacy, re, sys
from stanfordcorenlp import StanfordCoreNLP

ALLOWED_DIFFS = {'a', 'an', 'be', 'ing', 'who', 'which', 'that'
                 'is', 'are', 'am', 'was', 'were'}  # TODO add 'the'?
ALLOWED_DIFFS_CONTRA = {'not', 'no', "n't"}
PAT_AND_THERE_BE_NO = re.compile("[Tt]here (is|are) no.+and there (is|are) no.+")
BE_VERBS = {'be', 'is', 'are', 'were', 'was', 'am'}
PAT_NAIVE = re.compile("[Tt]here (is|are) no (.+?) (\w+ing\s?.*)")

SENTS = [

    # not solved:
    "There is no brown dog standing in the water",
    # TODO there is no ... AND there is no ... split into 2 parts? 2 premises?
    "There is no girl in white dancing",
    "There is no woman being kissed by a man",
    "There is no man running, jumping and kicking near a vending machine",
    "There is no boy playing outdoors and there is no man smiling",

    # sovled:
    # "There is no man eating food",
    # "There is no karate practitioner kicking at another man who is wearing protective boxing gloves",
    # "There is no man in a red uniform making a jump in a dirt bike race",
    # "There is no adult in the amphitheater talking to a boy",
    # "There is no brown dog running across the yard with a toy in its mouth",
    # "There is no little boy wearing a green soccer strip and running on the grass",
    # "There is no bmx biker jumping dirt ramps in front of a body of water",
    # "There is no woman putting on makeup",
    # "There are no racers running down a track",

]

def main():
    # test2()
    # test()
    there_be_no()

def there_be_no():
    corenlp = StanfordCoreNLP('http://localhost', port=9000, lang='en')
    # counter_double_there_be_no = 0

    fout = open("SICK/sick.there_be_no.trial.train.new_sent.txt", 'w')
    fout.write("{}\t{}\t{}\t{}\tcorrect\n".format("old", "dep_parse", "regex", "diff"))

    SENTS = [x.strip() for x in open("SICK/sick.there_be_no.trial.train.txt").readlines()]

    for sent in SENTS:
        sent = sent.strip()
        new_sent, new_sent2, diff = there_be_no_sent(sent, corenlp)

        fout.write("{}\t{}\t{}\t{}\t?\n".format(sent, new_sent, new_sent2, diff))

    fout.close()
    corenlp.close()
    # print("\n# double there be no:", counter_double_there_be_no)

def there_be_no_sent(sent, corenlp):
    """ get new_sent for one sent """
    # print('-' * 30)
    # print(sent)
    # if conjunction, then split to 2 sents
    if PAT_AND_THERE_BE_NO.match(sent):
        # print("double_there_be_no! TODO")
        # counter_double_there_be_no += 1
        new_sent = sent
        new_sent2 = sent
        diff = "same"
    else:
        dep_tree = build_dep_tree(sent, corenlp)
        new_sent = get_new_sent_dep(dep_tree)
        new_sent2 = get_new_sent_regex(sent)
        diff = "diff"
        if new_sent.strip() == new_sent2.strip(): diff = "same"
        
    return new_sent, new_sent2, diff

def get_new_sent_regex(sent):
    """
    naive method, match: there be no ... V+ing
    """
    m = PAT_NAIVE.match(sent)
    if m:
        # print()
        # print(m.group(2))  # brown dog
        # print(m.group(1))  # is
        # print(m.group(3))
        new_sent = " ".join(['No', m.group(2), m.group(1), m.group(3)])
    else:
        new_sent = sent
        # print("can't transform sent")

    return new_sent

def get_new_sent_dep(dep_tree):
    """ return new sent as a string 
    we want: no karate practitioner is kicking at another man
    """
    # ------------------------------
    #  get subject
    # ------------------------------
    #
    # 1. one-word noun: There is no man eating food
    # [('ROOT', 0, 2), ('expl', 2, 1), ('neg', 4, 3), ('nsubj', 2, 4), ('acl', 4, 5), ('dobj', 5, 6)]
    subj_parts_idx = []
    subj = {}
    subj_end_idx = {}   # subject ends at index i
    subj_counter = 0
    
    for i in dep_tree.idx2node:
        node = dep_tree.idx2node[i]
        if node.rel == 'nsubj':
            if node.token in {'who', 'which', 'that'}:
                subj_counter += 1  # maybe more than 1 subj
                subj_parts_idx = []
                continue  # exclude: there is no man *who* ...

            subj_parts_idx.append(node.idx)
            # print('\n-- head of subj:', node)
            # TODO KEY STEP
            for d in node.dependents:  # d is a node
                if d.rel == 'neg': continue  # d = "no"
                elif d.rel == 'acl': continue  # d = the clause: "eating food"
                elif d.rel == 'dep': continue  # d = smiling in "there is no man smiling
                elif d.rel == 'amod': subj_parts_idx.append(d.idx)
                elif d.rel == 'compound': subj_parts_idx.append(d.idx)
                else:
                    subj_parts_idx.append(d.idx)
                    # print(d, d.rel)

            if len(subj_parts_idx) == 1:
                subj[subj_counter] = dep_tree.idx2node[subj_parts_idx[0]].token
                subj_end_idx[subj_counter] = subj_parts_idx[0]

            # if subj longer than one word, then use the span from min to max
            else:
                subj[subj_counter] = dep_tree.tokens_i_to_j(min(subj_parts_idx), max(subj_parts_idx))
                subj_end_idx[subj_counter] = max(subj_parts_idx)

            subj_counter += 1  # maybe more than 1 subj
            subj_parts_idx = []

    # 2. JJ + NN: There is no karate practitioner kicking at another man...
    # 3. NN + PP: There is no man in a red uniform making a jump in a dirt bike race
    # 4. NN + RC: TODO
    # 5. AND: There is no boy playing outdoors and there is no man smiling

    # only one nsubj relation!
    if len(subj) == 0:
        # print("** no subj **")
        return "** no subj **"
    elif len(subj) > 1:
        # print("** more than 1 subj **")
        return "** more than 1 subj **"

    # ------------------------------
    # get the rest of the sentence, and be
    # ------------------------------
    # "There is no woman putting on makeup": find index of "putting"
    try:
        rest = dep_tree.tokens_i_to_j(i=subj_end_idx[0]+1, j=len(dep_tree.idx2node)-1)
    except KeyError:
        return "** can't transform **"
    # print("\nsubj no. {}".format(0))
    # print("subj:", subj[0])
    if dep_tree.idx2node[2].token not in BE_VERBS:
        return "** can't transform **"
    # print("be  :", dep_tree.idx2node[2].token)
    # print("rest:", rest)
    new_sent = "No {} {} {}".format(subj[0], dep_tree.idx2node[2].token, rest)
    # print("new sent:", new_sent)

    return new_sent

def build_dep_tree(sent, corenlp):
    """ build dep tree from input sent_str """
    # idx=0 is root in dep parse
    idx2tok = {idx + 1: word for idx, word in enumerate(corenlp.word_tokenize(sent))}
    idx2tok[0] = "ROOT"

    # build dep_tree
    dep_tree = DepTree()
    for dep_trip in corenlp.dependency_parse(sent):
        rel, head_idx, dependent_idx = dep_trip[0], dep_trip[1], dep_trip[2]
        # head, dependent both already in tree
        if (head_idx in dep_tree.idx2node) and (dependent_idx in dep_tree.idx2node):
            head_node = dep_tree.idx2node[head_idx]
            dependent_node = dep_tree.idx2node[dependent_idx]

        elif (head_idx in dep_tree.idx2node) and (dependent_idx not in dep_tree.idx2node):
            head_node = dep_tree.idx2node[head_idx]
            dependent_node = DepNode(dependent_idx, idx2tok[dependent_idx])
            dep_tree.idx2node[dependent_idx] = dependent_node

        elif (head_idx not in dep_tree.idx2node) and (dependent_idx in dep_tree.idx2node):
            head_node = DepNode(head_idx, idx2tok[head_idx])
            dependent_node = dep_tree.idx2node[dependent_idx]
            dep_tree.idx2node[head_idx] = head_node

        else:  # neither in idx2node
            head_node = DepNode(head_idx, idx2tok[head_idx])
            dependent_node = DepNode(dependent_idx, idx2tok[dependent_idx])
            dep_tree.idx2node[head_idx] = head_node
            dep_tree.idx2node[dependent_idx] = dependent_node

        build_dep_rel(head_node, dependent_node, rel)
    return dep_tree

def build_dep_rel(head_node, dependent_node, rel):
    head_node.dependents.append(dependent_node)
    dependent_node.head = head_node
    dependent_node.rel = rel

class DepTree:
    def __init__(self, root=None):
        self.root = root
        self.idx2node = {}

    def tokens_i_to_j(self, i, j):
        """ return tokens from idx=i to j inclusive"""
        nodes = [ self.idx2node[x] for x in range(i, (j+1)) ]
        return " ".join([ node.token for node in nodes ])

class DepNode:
    def __init__(self, idx=None, token=None, dependents=None, head=None, rel=None, pos=None):
        self.idx = idx
        self.token = token
        if dependents: self.dependents = dependents
        else: self.dependents = []
        self.head = head   # a node
        self.rel = rel     # relation to the head
        self.rels_deps = []  # relations to dependents TODO maybe a dict?
        self.pos = pos

    def get_children(self):
        pass

    def __str__(self): return "DepNode_{} {}".format(self.idx, self.token)

    def __repr__(self): return self.__str__()

# def test2():
#     nlp = spacy.load('en')

#     for sent in SENTS:
#         # print(sent)
#         # continue
#         there_be_no(sent, nlp)
#     pass

'''
def there_be_no(sent_str, nlp):
    """
    there be no man dancing -> no man be dancing
    """
    # use spacy to get dep parse
    # or: use CoreNLP to get dep parser
    doc = nlp(sent_str)

    # find subject

    subj = get_subj(doc, nlp)

    # convertion
    pass
'''

"""
def get_subj(doc, nlp):
    print('\n')
    myformat = "{:15}\t{:10}\t{:15}\t{:6}\t{}"
    for sent in doc.sents:
        print(sent.root)
        for child in sent.root.children:
            if child.dep_ == "attr":  # the subject always has attr
                print(child.text, child.i)

                print("subtree:", [t.text for t in child.subtree])

        print()
        print(myformat.format("|tok|", "|dep_rel|", "|head|", "|head_pos|", "|children|"))
        for token in sent:
            print(myformat.format(token.text, token.dep_, token.head.text, token.head.pos_,
                  [child for child in token.children]))
            #### types of subjects ####
            # 1. one-word noun: There is no man eating food
            # 2. JJ + NN: There is no karate practitioner kicking at another man...
            # 3. NN + PP: There is no man in a red uniform making a jump in a dirt bike race
            # 4. NN + RC: TODO

            # OUTPUT of SpaCy
            # |tok|          	|dep_rel| 	|head|         	|head_pos|	|children|
            # There          	expl      	is             	VERB  	[]
            # is             	ROOT      	is             	VERB  	[There, man]
            # no             	det       	man            	NOUN  	[]
            # man            	attr      	is             	VERB  	[no, in, making]
            # in             	prep      	man            	NOUN  	[uniform]
            # a              	det       	uniform        	NOUN  	[]
            # red            	amod      	uniform        	NOUN  	[]
            # uniform        	pobj      	in             	ADP   	[a, red]
            #
            # There          	expl      	is             	VERB  	[]
            # is             	ROOT      	is             	VERB  	[There, adult, talking]
            # no             	det       	adult          	NOUN  	[]
            # adult          	attr      	is             	VERB  	[no, in]
            # in             	prep      	adult          	NOUN  	[amphitheater]
            # the            	det       	amphitheater   	NOUN  	[]
            # amphitheater   	pobj      	in             	ADP   	[the]
            # talking        	acl       	is             	VERB  	[to]
            # to             	prep      	talking        	VERB  	[boy]
            # a              	det       	boy            	NOUN  	[]
            # boy            	pobj      	to             	ADP   	[a]

            #

    return None
"""

def test():
    test_tuples = [
        ("A young girl in a swimming suite is jumping on the beach",
         "a young girl in swimming suite is jump on the beach"),
        ("A man is rock climbing, pausing and calculating the route",
         "A man is rock climbing, stopping and calculating the route")
    ]
    for pair in test_tuples:
        print(similar_str(pair[0], pair[1]))

def similar_str(s1, s2):
    s1_list = s1.lower().replace("ing", " ing ").replace('-', ' ').split()
    s2_list = s2.lower().replace("ing", " ing ").replace('-', ' ').split()

    diff = difflib.context_diff(s1_list, s2_list, lineterm="")

    for i, s in enumerate(diff):
        if s[0:2] in {'- ', '+ ', '! '}:
            if s[2:] not in ALLOWED_DIFFS: return False

    return True

def similar_str_contra(s1, s2):
    s1_list = s1.lower().replace("ing", " ing ").replace('-', ' ').split()
    s2_list = s2.lower().replace("ing", " ing ").replace('-', ' ').split()

    diff = difflib.context_diff(s1_list, s2_list, lineterm="")

    num_negation = 0

    for i, s in enumerate(diff):
        if s[0:2] in {'- ', '+ ', '! '}:
            if s[2:] in ALLOWED_DIFFS: continue
            if s[2:] in ALLOWED_DIFFS_CONTRA: num_negation += 1
            else: return False

    if num_negation == 1: return True
    else: return False


# ---------------------------------

def P_idx(sick2uniq, sick_id):
    """ return idx of P in sick2uniq """
    return sick2uniq[sick_id]["P"]


def H_idx(sick2uniq, sick_id):
    """ return idx of H in sick2uniq """
    return sick2uniq[sick_id]["H"]
# ---------------------------------


def quantifier_node(word):
    node_lemma = word
    if word.upper() in {"A", "ONE", "EVERY", "EACH", "THE", "SOME"}: number = "sg"
    else: number = "pl"
    return LeafNode(depth=0, cat=Cat('NP/N', word=node_lemma),
                    chunk=None, entity=None,
                    lemma=node_lemma, pos='DT', span=None, start=None,
                    word=node_lemma,impType=None,fixed=False,note=None,
                    number=number)

# ---------------------------------

node_every = quantifier_node('every')
node_all = quantifier_node('all')
node_each = quantifier_node('each')
node_some = quantifier_node('some')
node_a = quantifier_node('a')
node_an = quantifier_node('an')
node_many = quantifier_node('many')
node_several = quantifier_node('several')
node_few = quantifier_node('few')
node_most = quantifier_node('most')
node_the = quantifier_node('the')
node_at_least_N = quantifier_node('at-least-N')

# ---------------------------------

tree_every = CCGtree(TermNode=(quantifier_node('every')))
tree_all = CCGtree(TermNode=(quantifier_node('all')))
tree_each = CCGtree(TermNode=(quantifier_node('each')))
tree_some = CCGtree(TermNode=(quantifier_node('some')))
tree_a = CCGtree(TermNode=(quantifier_node('a')))
tree_an = CCGtree(TermNode=(quantifier_node('an')))
tree_many = CCGtree(TermNode=(quantifier_node('many')))
tree_several = CCGtree(TermNode=(quantifier_node('several')))
tree_few = CCGtree(TermNode=(quantifier_node('few')))
tree_most = CCGtree(TermNode=(quantifier_node('most')))
tree_the = CCGtree(TermNode=(quantifier_node('the')))
tree_at_least_N = CCGtree(TermNode=(quantifier_node('at-least-N')))

# ---------------------------------

node_X = LeafNode(depth=0, cat=Cat('N', word="X"),
                  chunk=None, entity=None,
                  lemma="X", pos='N', span=None, start=None,
                  word="X")

node_X_bar = LeafNode(depth=0, cat=Cat('N', word="X_bar"),
                  chunk=None, entity=None,
                  lemma="X_bar", pos='N', span=None, start=None,
                  word="X_bar")

node_Y = LeafNode(depth=0, cat=Cat('N', word="Y"),
                  chunk=None, entity=None,
                  lemma="Y", pos='N', span=None, start=None,
                  word="Y")

node_Z = LeafNode(depth=0, cat=Cat('N', word="Z"),
                  chunk=None, entity=None,
                  lemma="Z", pos='N', span=None, start=None,
                  word="Z")


def make_leafnode(cat, word, lemma, pos, number=None):
    return getMono.LeafNode(depth=0, cat=getMono.Cat(cat, word=word),
                         chunk=None, entity=None,
                         lemma=lemma, pos=pos, span=None, start=None,
                         word=word, number=number)


def make_nontermnode(cat, ruleType, number=None):
    ans = getMono.NonTermNode(depth=0, cat=getMono.Cat(cat, word=None),
                               ruleType=ruleType, wholeStr="", number=number)
    ans.assignWholeStr()
    return ans


def save_sick_id_2_uniq_id_mapping():
    """  mapping:  sick_id, e.g., 5 ==>
    dict{"P": 30, "H": 33} where 30 and 33 are indices in sick_uniq.raw.txt """
    fout = open('sick_id_2_uniq_id.txt', 'w')
    fout.write("sick_id\tP_id_in_uniq\tH_id_in_uniq\n")
    # read in uniq
    uniq = open('sick_uniq.raw.txt').readlines()
    uniq = [line.strip() for line in uniq]
    # read in SICK.tsv
    # sick = {}  # {5: {"P": sent_str, "H": sent_str} }
    with open("SICK/SICK.tsv") as f:
        text = csv.reader(f, delimiter="\t")
        for line in text:
            if line[0] == "pair_ID":
                print("first line")
                continue
            assert len(line) == 12
            premise = line[1].strip()
            hypothesis = line[2].strip()
            sick_id = line[0]
            assert premise in uniq
            assert hypothesis in uniq
            fout.write("{}\t{}\t{}\n".format(sick_id,
                                             uniq.index(premise), uniq.index(hypothesis)))
    fout.close()

def sick_id_2_uniq_id():
    """ input: sick_id, e.g., 5, starting from 1!
    output: dict{"P": 30, "H": 33}
    where 30 and 33 are indices in sick_uniq.raw.txt, starting from 0! """
    sick2uniq = {}
    with open("sick_id_2_uniq_id.txt") as f:
        for line in f:
            if line.startswith("sick_id"): continue  # first line
            line_split = line.split('\t')
            assert  len(line_split) == 3
            sick_id, P_id_in_uniq, H_id_in_uniq = line_split[0], line_split[1], line_split[2]
            sick2uniq[int(sick_id)] = {"P": int(P_id_in_uniq), "H": int(H_id_in_uniq)}
    return sick2uniq


def set_parent_children(parent_node, children_list):
    """ set relations between parent and children,
    parent: a node, children: a list """
    parent_node.children = children_list
    for child in children_list: child.parent = parent_node
    if len(children_list) == 2:
        children_list[0].sisters = [children_list[1]]
        children_list[1].sisters = [children_list[0]]
    parent_node.assignWholeStr()


def find_head_noun(node_det, ccg_tree):
    """ find the head noun of a determiner in a tree """
    # - configuration 1: the brown bulldog who every brown mammal moved-towards
    #            bulldog   RC
    #           ---------------
    #               N       N\N
    #       brown -------------
    #         N/N       N
    #  the   --------------
    #   NP/N       N = sister
    # - configuration 2: the brown dog
    # - configuration 3: the dog
    # - configuration 4: a good poodle who was happy waltzed

    # ----------------------------------------------
    # solution 1: find the first leafnode that is of category N, starting
    # from the sister of the determiner
    # -- this does NOT work for configuration 4, b/c (good poodle) and (RC)
    #    both have two children

    # sister = node_det.sisters[0]
    # head_noun = sister
    # while not head_noun_test(head_noun):
    #     old_head_noun = head_noun
    #     if len(head_noun.children) == 0:
    #         eprint("something wrong finding head noun, no children for {}".format(head_noun))
    #         eprint(ccg_tree.tree_str())
    #         exit()
    #     for child in head_noun.children:
    #         if head_noun_test(child):
    #             head_noun = child  # found head_noun
    #             break
    #         if len(child.children) != 0:
    #             head_noun = child  # next, explore the child which has children
    #     if old_head_noun == head_noun:
    #         eprint("\n!! something wrong finding head noun")
    #         eprint("old head noun:", old_head_noun)
    #         eprint("new head noun:", head_noun)
    #         eprint(ccg_tree.tree_str())
    #         ccg_tree.printTree(stream=sys.stderr)
    #         eprint("\n")
    #         exit()
    # if head_noun is None:
    #     eprint("head noun for det **{}** is None".format(node_det))
    #     eprint(ccg_tree.tree_str())
    #     exit()

    # ------------------------------
    # solution 2: find the *highest* leafnode (N) under the NP
    depth_min = -1
    head_noun = None
    nodes_under_NP = [ node_det.sisters[0] ]
    while nodes_under_NP:
        poped_node = nodes_under_NP.pop()
        if poped_node.children:  # non term node
            nodes_under_NP.extend(poped_node.children)
        else:  # leaf node
            if poped_node.cat.typeWOfeats == "N" and poped_node.depth > depth_min:
                head_noun = poped_node

    if head_noun is None:
        eprint("\n!! something wrong finding head noun")
        eprint(ccg_tree.tree_str())
        ccg_tree.printTree(stream=sys.stderr)
        eprint("\n")
        exit()

    return head_noun


def head_noun_test(node):
    return node.cat.typeWOfeats == "N" and len(node.children) == 0



# ---------------------------
# helper functions

def same_ani_noun(leaf_node_1, leaf_node_2):
    """ check if they are the same animate noun: animal = animals """
    if abs(len(leaf_node_1.lemma) - len(leaf_node_2.lemma)) > 2: return False
    idx = min([len(leaf_node_1.lemma), len(leaf_node_2.lemma)])  # could be box vs boxes
    return leaf_node_1.lemma[:idx] == leaf_node_2.lemma[:idx]

def assign_num_to_be(ccg_tree):
    for ln in ccg_tree.leafNodes:
        if ln.lemma in {"be", "was", "were"}: assign_num_to_be_helper(ln)

def assign_num_to_be_helper(node_be):
    if len(node_be.parent.sisters) == 0:
        return
    # case 1: every dog who was sad ran
    #                 was  sad
    #                ----------
    #             who     was-sad
    #            ------------------
    #        dog        who-was-sad
    if node_be.parent.sisters[0].wholeStr == "WHO":
        if node_be.parent.parent.sisters[0].number == "sg":
            node_be.assign_new_word("was", "sg")
        elif node_be.parent.parent.sisters[0].number == "pl":
            node_be.assign_new_word("were", "Pl")
        else:
            print("something went wrong when assigning number to 'be'")
            exit()

    # case 2: every dog was sad
    # every dog            was sad
    # -----------        -----------
    #   every-dog          was-sad
    elif node_be.parent.sisters[0].cat.typeWOfeats == "NP":
        # print("NP:", node_be.parent.sisters[0])
        if node_be.parent.sisters[0].number == "sg":
            node_be.assign_new_word("was", "sg")
        elif node_be.parent.sisters[0].number == "pl":
            node_be.assign_new_word("were", "pl")
        else:
            print("something went wrong when assigning number to 'be'")
            exit()

    else:
        print("something went wrong when assigning number to 'be'")
        exit()


################################
# added  from generate_wordlist.py
def at_least_n(mynum=None):
    """ if mynum is specified, then n = mynum, can be: 3,4,5,6 """
    #  at-least    n
    # (NP/N)/(N/N)    N/N
    # --------------------  fa
    #      NP/N = node_new
    # pos of at-least is "DET"
    node_new = make_nontermnode(cat=r"NP/N", ruleType="fa")
    node_new.number = "pl"
    at_least = make_leafnode(cat=r"(NP/N)/(N/N)", word="at-least", lemma="at-least", pos="DET")
    at_least.note = "at-least"
    if mynum is None: set_parent_children(parent_node=node_new, children_list=[at_least, num()])
    else:
        n = make_leafnode(cat=r"N/N", word=str(mynum), lemma=str(mynum), pos="CD")
        set_parent_children(parent_node=node_new, children_list=[at_least, n])
    return node_new
    
node_at_least_3 = at_least_n(3)
node_at_least_several = at_least_n("several")


if __name__ == '__main__':
    main()

