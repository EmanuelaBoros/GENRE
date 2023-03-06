from genre.trie import Trie, MarisaTrie
import pickle

with open("../data/lang_title2wikidataID-normalized_with_redirect.pkl", "rb") as f:
    lang_title2wikidataID = pickle.load(f)

# fast but memory inefficient prefix tree (trie) -- it is implemented with nested python `dict`
# NOTE: loading this map may take up to 10 minutes and occupy a lot of RAM!
# with open("../data/titles_lang_all105_trie_with_redirect.pkl", "rb") as f:
#     trie = Trie.load_from_dict(pickle.load(f))

# memory efficient but slower prefix tree (trie) -- it is implemented with `marisa_trie`
with open("../data/titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
    trie = pickle.load(f)

# for pytorch/fairseq
from genre.fairseq_model import mGENRE

import pickle
import sys

from genre.utils import get_entity_spans_fairseq as get_entity_spans
from genre.fairseq_model import mGENRE
from genre.utils import get_markdown


if __name__ == "__main__":
    model_path = "../models/fairseq_multilingual_entity_disambiguation"
    # dict_path = "data/mention_to_candidates_dict.pkl"
    trie_path = "../data/titles_lang_all105_marisa_trie_with_redirect.pkl"

    print('Loading {}'.format(model_path))
    model = mGENRE.from_pretrained(model_path).eval()
    print('Loading {}'.format(trie_path))
    with open(trie_path, "rb") as f:
        mention_trie = pickle.load(f)
    # with open(dict_path, "rb") as f:
    #     mention_to_candidates_dict = pickle.load(f)

    text = """Home Depot CEO Nardelli quits Home-improvement retailer's chief executive had been criticized over pay ATLANTA - Bob Nardelli abruptly resigned Wednesday as chairman and chief executive of The Home Depot Inc. after a six-year tenure that saw the world’s largest home improvement store chain post big profits but left investors disheartened by poor stock performance. Nardelli has also been under fire by investors for his hefty pay and is leaving with a severance package valued at about $210 million. He became CEO in December 2000 after being passed over for the top job at General Electric Co., where Nardelli had been a senior executive. Home Depot said Nardelli was being replaced by Frank Blake, its vice chairman, effective immediately. Blake’s appointment is permanent, Home Depot spokesman Jerry Shields said. What he will be paid was not immediately disclosed, Shields said. The company declined to make Blake available for comment, and a message left for Nardelli with his secretary was not immediately returned. Before Wednesday’s news, Home Depot’s stock had been down more than 3 percent on a split-adjusted basis since Nardelli took over. Nardelli’s sudden departure was stunning in that he told The Associated Press as recently as Sept. 1 that he had no intention of leaving, and a key director also said that the board was pleased with Nardelli despite the uproar by some investors. Asked in that interview if he had thought of hanging up his orange apron and leaving Home Depot, Nardelli said unequivocally that he hadn’t. Asked what he thought he would be doing 10 years from now, Nardelli said, “Selling hammers.” For The Home Depot? “Absolutely,” he said at the time. Home Depot said Nardelli’s decision to resign was by mutual agreement with the Atlanta-based company. “We are very grateful to Bob for his strong leadership of The Home Depot over the past six years. Under Bob’s tenure, the company made significant and necessary investments that greatly improved the company’s infrastructure and operations, expanded our markets to include wholesale distribution and new geographies, and undertook key strategic initiatives to strengthen the company’s foundation for the future,” Home Depot’s board said in a statement. Nardelli was a nuts-and-bolts leader, a former college football player and friend of President Bush. He helped increase revenue and profits at Home Depot and increase the number of stores the company operates to more than 2,000. Home Depot’s earnings per share have increased by approximately 150 percent over the last five years."""

    sentences = [text]

    results = model.sample(
        sentences,
        beam=10,
        prefix_allowed_tokens_fn=lambda batch_id, sent: [
            e for e in trie.get(sent.tolist())
            if e < len(model.task.target_dictionary)
            # for huggingface/transformers
            # if e < len(model2.tokenizer) - 1
        ],
    )

    entity_spans = get_entity_spans(
        model,
        sentences,
        mention_trie=mention_trie,
        mention_to_candidates_dict=model.task.target_dictionary
    )
    import pdb;pdb.set_trace()
    markdown = get_markdown(sentences, entity_spans)[0]
    print(markdown)
