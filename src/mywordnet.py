#!/usr/bin/env python3
"""
wordnet module

pos can be:
{NN: "n", VB: "v", JJ: "a", RB: "r"}
"NN", "VB", "JJ", "RB"

what we need:

1. for every pair of sentences, list_nouns, list_verbs, etc.
for every two nouns i, j in list_nouns: whether i < j, i = j, i > j, or no relation.
same for other lists

2. for every noun, verb, adv, adj, call it i,
we want i.hypernyms(level=1), i.hypernyms(level=2) ... level n
the same for i.hyponyms
also: synonyms, antonyms

3. a frequency dictionary (of BNC, gigaword or any large corpus)

Hai Hu
"""

from pattern.en import wordnet
from wordfreq import zipf_frequency
from infer import Knowledge
import sick_ans
from utils import P_idx, H_idx
from getMono import CCGtrees, CCGtree, eprint
import itertools, sys, os
from copy import deepcopy
import spacy

def main():
    if '-s' in sys.argv:
        save_cache()
    elif '-t' in sys.argv:
        test()
    else:
        eprint('required arg: -s/-t')

def test():
    word = 'walk'
    pos = 'VB'

    print(wordnet.synsets(word, pos))

    return


    hypers = get_words(word, pos, 'hyper', True, 10)
    # for hyper in hypers:
    #     print(hyper, hypers[hyper])
    print(hypers.keys())
    # print('fruit' in hypers)

    hypos = get_words(word, pos, 'hypo', True, 1)
    # for hypo in hypos:
    #     print(hypo, hypos[hypo])
    print(hypos.keys())

    print()
    print(get_synonyms('big', 'JJ'))
    print(get_antonyms('big', 'JJ'))
    print(get_similar('big', 'JJ'))
    print()
    print(get_synonyms('man', 'NN'))
    print(get_antonyms('man', 'NN'))
    print(get_similar('man', 'NN'))
    print()

    print(wordnet.synsets('go', pos='VB')[0])
    print(wordnet.synsets('nice', pos='JJ')[0])
    print(wordnet.synsets('well', pos='RB')[0])
    print(wordnet.synsets('musical instrument', pos="NN")[0])

def save_cache():
    """
    for each noun in train[:500] + trials
    save the 5 most frequent hypers and hypos
    """
    sick_ids = sick_ans.ids_trial_E[:10]  # ids_trial_E_C
    sick_ids = [211, 1412, 1495, 2557, 2829, 2898, 3250, 4015, 4066, 4135,
                4342, 4360, 4661, 4916, 5030, 5113, 5498, 5806, 6756]
    # 4916=guitar, musical instrument
    # 4006 throw, hurl
    # 6756 large big
    sick_ids = [5498, 5806, 4916, 4066, 6756]  # got all these

    trees = CCGtrees("sick_uniq.raw.tok.preprocess.log")
    trees.readEasyccgStr("sick_uniq.raw.easyccg.parsed.txt")

    for id_to_solve in sick_ids:
        eprint("-"*50)
        print("sick", id_to_solve)
        P = trees.build_one_tree(H_idx(sick_ans.sick2uniq, id_to_solve), "easyccg", use_lemma=False)
        H = trees.build_one_tree(P_idx(sick_ans.sick2uniq, id_to_solve), "easyccg", use_lemma=False)
        eprint("P:", end="")
        P.printSent_raw_no_pol(stream=sys.stderr)
        eprint("H:", end="")
        H.printSent_raw_no_pol(stream=sys.stderr)

        k = Knowledge()
        k.update_word_lists(P)  # nouns, subSecAdj, etc.
        k.update_word_lists(H)
        # k.update_modifier()

        assign_all_relations_wordnet(k)

        # k.print_knowledge()

def add_alternations(sent_id, nlp, k):
    """  for each word in tokenized_sent, we first use JIGSAW to disambiguate,
    then find all its alternations, add to k.frags
    sent_id is the idx of the sent in uniq_sent

    A a o a U
    biker biker n biker U
    with with o with U
    a a o a U
    blue blue a blue a00370869
    jacket jacket n jacket n03589791
    black black a black a00392812
    pants pant n pant n07388706
    and and o and U
    a a o a U
    white white a white a00393105
    helmet helmet n helmet n03513376
    is is v is U
    driving drive v drive v01930874
    recklessly reckless r recklessly r00354861
    on on o on U
    dirt dirt n dirt n14844693
    and and o and U
    people peopl n people n07942152
    watch watch v watch v02150510
    """
    # read in /media/hai/G/tools/JIGSAW/test/sent_id.txt.jigsaw
    with open("/media/hai/G/tools/JIGSAW/test/" + str(sent_id) + ".txt.jigsaw") as f:
        for line in f:
            line_split = line.split()
            word, pos, wordnet_id = line_split[0], line_split[2], line_split[-1]
            # get alternations
            if pos != 'o':
                wordnet_id = int(wordnet_id[1:])
                alternations = get_alternations(word, pos, wordnet_id, nlp)
    # todo
    pass

def get_alternations(word, pos, synset_id, nlp, verbose=False):
    """ return all alternations """
    alternations = {}  # {word_str : simi_score}
    # find all alternations of walk
    for ss in wordnet.synsets(word, pos):
        if int(ss.id) != int(synset_id): continue
        for hyper in ss.hypernyms():
            for hypo in hyper.hyponyms():
                if hypo.pos != pos: continue  # only want the same pos
                for synonym in hypo.synonyms:
                    if synonym.lower() == word.lower(): continue  # don't want ss here
                    if synonym not in alternations:
                        simi_score = nlp(word).similarity(nlp(synonym))
                        if simi_score > 0:
                            #                             print(word, synonym, simi_score)
                            alternations[synonym] = simi_score
    if verbose: print('found {} alternations'.format(len(alternations)))
    return alternations

def assign_all_relations_wordnet(k):
    # NP and NN, musical instrument
    assign_relations([k.nouns], k, pos="NN", depth=10)   # was: [k.nouns, k.NPs]
    assign_relations([k.subsecAdj], k, pos="JJ", depth=10)
    assign_relations([k.verbs], k, pos="VB", depth=10)
    assign_relations([k.advs], k, pos="RB", depth=10)

def assign_relations(word_lists, k, pos, depth):
    """ note that k.nouns = { wholeStr: {type1: node, type2: node} }
     so we need to make sure the two compared nodes are of the same type """
    full_dict = {}
    for word_list in word_lists: full_dict.update(word_list)
    words = sorted(full_dict.keys())
    for idx_pair in itertools.combinations(range(len(full_dict)), 2):
        # print(idx_pair)
        word1_wholeStr, word2_wholeStr = words[idx_pair[0]], words[idx_pair[1]]
        node1_dict = full_dict[word1_wholeStr]  # a dict: {type1: node1, type2: node2}
        node2_dict = full_dict[word2_wholeStr]

        # first find the relation between word1_wholeStr and word2_wholeStr
        word1_lower, word2_lower = word1_wholeStr.lower(), word2_wholeStr.lower()
        rel = find_relation(word1_lower, word2_lower, pos, depth)
        if rel:
            add_relation_wordnet(node1_dict, node2_dict, k, rel)

def find_relation(word1_lower, word2_lower, pos, depth):
    """ return: hypernym, hyponym, synonym, antonym, None """
    if word1_lower in get_words(word2_lower, pos, 'hyper', recursive=True, depth=depth):
        return 'hypernym'  # word1 > word2
    elif word1_lower in get_words(word2_lower, pos, 'hypo', recursive=True, depth=depth):
        return 'hyponym'  # word1 < word2
    elif word1_lower in get_synonyms(word2_lower, pos):
        return 'synonym'
    elif word1_lower in get_antonyms(word2_lower, pos):
        return 'antonym'
    elif word1_lower in get_similar(word2_lower, pos):
        return 'similar'
    return None

def add_relation_wordnet(node1_dict, node2_dict, k, relation):
    """  node1_dict: {type1: node1, type2: node2}  LeafNode, NonTermNode
    relation = hypernym hyponym synonym antonym similar
    """
    for node1 in node1_dict.values():
        for node2 in node2_dict.values():
            if node1.cat.typeWOfeats == node2.cat.typeWOfeats:
                node1 = deepcopy(node1)
                node2 = deepcopy(node2)
                print('-- relation: {:10} {}:{} {}:{}'.format(relation, node1.wholeStr, node2.wholeStr,
                                                        node1.cat.typeWOfeats, node2.cat.typeWOfeats))
                if relation == 'hyponym': k.add_pair((node1, node2), True)
                elif relation == 'hypernym': k.add_pair((node2, node1), True)
                elif relation == 'synonym': k.add_equal_pair(node1, node2, True)
                elif relation == 'antonym': k.add_antonym_pair(node1, node2, True)  # TODO !!!
                elif relation == 'similar': k.add_equal_pair(node1, node2, True) # TODO
                else:
                    print("something wrong in add_relation_wordnet")
                    exit()

def get_words(word, pos, hyper_hypo, recursive=False, depth=None):
    """ return a dict { str.lower : zipf_freq.log }
    if depth = NOne, depth = 1
    """
    ans = {}  # a dictioinary
    # find out which synset for the word
    for ss in wordnet.synsets(word, pos):
        # if ss.pos.startswith(pos.upper()):  # this is what we want!
        if hyper_hypo == 'hyper':
            for x in ss.hypernyms(recursive, depth):
                ans.update({syn.replace('_', ' ') : zipf_frequency(syn.replace('_', ' '), 'en') \
                            for syn in x.synonyms})
        elif hyper_hypo == 'hypo':
            for x in ss.hyponyms(recursive, depth):
                ans.update({syn.replace('_', ' ') : zipf_frequency(syn.replace('_', ' '), 'en') \
                            for syn in x.synonyms})
        # break  # only use the first synset, assuming it's the most common one
    return ans

def get_synonyms(word, pos):
    """ return a set of strings, lowercase unless proper noun """
    ans = set()
    for ss in wordnet.synsets(word, pos):
        synonyms = [synonym.replace('_', ' ') for synonym in ss.synonyms]
        # print(synonyms)
        ans.update(synonyms)
    return ans

def get_antonyms(word, pos):
    """ return a set of strings, lowercase unless proper noun """
    ans = set()
    for ss in wordnet.synsets(word, pos):
        if ss.antonym:  # a list of synsets
            for ss1 in ss.antonym:
                antonyms = [synonym.replace('_', ' ') for synonym in ss1.synonyms]
                # print(antonyms)
                ans.update(antonyms)
    return ans

def get_similar(word, pos):
    """ return a set of strings, lowercase unless proper noun """
    ans = set()
    for ss in wordnet.synsets(word, pos):
        if ss.similar():  # a list of synsets
            for ss1 in ss.similar():
                similars = [synonym.replace('_', ' ') for synonym in ss1.synonyms]
                # print(similars)
                ans.update(similars)
    return ans

if __name__ == '__main__':
    main()

