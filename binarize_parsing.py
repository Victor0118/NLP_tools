from pycorenlp import StanfordCoreNLP
from nltk import Tree
from functools import reduce
import re

# java -Djava.io.tmpdir=tmp -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000 -port 9012
nlp = StanfordCoreNLP('http://localhost:9012')

def binarize(tree):
    """
    Recursively turn a tree into a binary tree.
    """
    if isinstance(tree, str):
        return tree
    elif len(tree) == 1:
        return binarize(tree[0])
    else:
#         label = tree.label()
#         return reduce(lambda x, y: Tree(label, (binarize(x), binarize(y))), tree)
#         return reduce(lambda x, y: (binarize(x), binarize(y)), tree)
        return reduce(lambda x, y: " ( " + binarize(x) + " " + binarize(y) + " ) ", tree)

def transfer(sent):
    # input: what is your review of hidden figures ( 2016 movie )
    # output: ( what ( is ( ( your review ) ( of ( ( hidden figures ) ( -LRB- ( ( 2016 movie ) -RRB- ) ) ) ) ) ) ) 
    res = nlp.annotate(sent,
       properties={
           'annotators': 'parse',
           'outputFormat': 'json',
           'timeout': 1000,
       })

    t = Tree.fromstring(res['sentences'][0]['parse'])
    bt = binarize(t)
    return re.sub(' +', ' ',bt).strip()


transfer("what is your review of hidden figures ( 2016 movie )")
# '( what ( is ( ( your review ) ( of ( ( hidden figures ) ( ( -LRB- ( 2016 movie ) ) -RRB- ) ) ) ) ) )'
