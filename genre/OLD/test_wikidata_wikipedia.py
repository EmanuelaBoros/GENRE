from helper_pickle import pickle_load
from trie import Trie, MarisaTrie
lang_title2wikidataID_path = "../data/lang_title2wikidataID-normalized_with_redirect.pkl"

lang_title2wikidataID = pickle_load(lang_title2wikidataID_path, verbose=True)

import pdb;pdb.set_trace()