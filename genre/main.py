# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import pickle
import re
# for pytorch/fairseq
from fairseq_model import mGENRE
from model import Model
import jsonlines
import pandas
from utils import chunk_it, get_wikidata_ids
from tqdm.auto import tqdm, trange
from data import _read_conll, get_entities
import pickle
# from wikidata.client import Client
from tqdm import tqdm


# fast but memory inefficient prefix tree (trie) -- it is implemented with nested python `dict`
# NOTE: loading this map may take up to 10 minutes and occupy a lot of RAM!
# with open("../data/titles_lang_all105_trie_with_redirect.pkl", "rb") as f:
#     trie = Trie.load_from_dict(pickle.load(f))

# memory efficient but slower prefix tree (trie) -- it is implemented with `marisa_trie`
# with open("../data/titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
#     trie = pickle.load(f)

# trie_path = "../data/titles_lang_all105_marisa_trie_with_redirect.pkl"
trie_path = "../data/titles_lang_all105_trie_with_redirect.pkl"
model_path = "../models/fairseq_multilingual_entity_disambiguation"
lang_title2wikidataID_path = "../data/lang_title2wikidataID-normalized_with_redirect.pkl"

model = Model(model_name=model_path,
              mention_trie=trie_path,
              lang_title2wikidataID=lang_title2wikidataID_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
    )
    parser.add_argument(
        "--base_wikidata",
        type=str,
        help="Base folder with Wikidata data.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    args, _ = parser.parse_known_args()

    logging.basicConfig(level=logging.DEBUG)

    for root, dirs, files in os.walk(args.input_dir, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            print(filename)

            kilt_dataset = []
            if ("test" in filename) and ('.tsv' in filename) \
                    and ('masked' not in filename) \
                    and ('results' not in filename) \
                    and ('bart' not in filename):

                with open(filename, 'r') as f:
                    lines = f.readlines()

                headers = [
                    'raw_words', 'target', 'link'
                ]
                # TODO: This needs to be changed if the data format is different or the
                # order of the elements in the file is different
                indexes = list(range(10))  # -3 is for EL
                columns = [
                    "TOKEN",
                    "NE-COARSE-LIT",
                    "NE-COARSE-METO",
                    "NE-FINE-LIT",
                    "NE-FINE-METO",
                    "NE-FINE-COMP",
                    "NE-NESTED",
                    "NEL-LIT",
                    "NEL-METO",
                    "MISC"]
                if not isinstance(headers, (list, tuple)):
                    raise TypeError(
                        'invalid headers: {}, should be list of strings'.format(headers))
                phrases = _read_conll(
                    filename,
                    encoding='utf-8',
                    sep='\t',
                    indexes=indexes,
                    dropna=True)

                # entity = client.get('Q7344037', load=True)

                sentences = []
                with open(filename.replace('.tsv', '_results.tsv'), 'w') as f:
                    f.write('\t'.join(columns) + '\n')
                    for phrase in tqdm(phrases, total=len(phrases)):

                        # [('R . Ellis', 'pers', 'Q7344037'), ('the Cambridge Journal of Philology', 'work', 'NIL'),
                        # ('Vol . IV', 'scope', 'NIL'), ('A . Nauck', 'pers', 'NIL'), ('Leipzig', 'loc', 'NIL'), ('1856',
                        # 'date', 'NIL')]

                        idx, phrase = phrase
                        tokens, entity_tags, link_tags = phrase[0], phrase[1], phrase[-3]
                        entities = get_entities(tokens, entity_tags, link_tags)

                        # [('R . Ellis', 'pers', 'Q7344037', [12, 13, 14])
                        pos_qid, pos_ner = {}, {}
                        for entity in entities:
                            meta = {
                                "left_context": ' '.join(tokens[:entity[-1][0]]),
                                "right_context": ' '.join(tokens[entity[-1][-1] + 1:]),
                                "mention": entity[0],
                                "label_title": entity,
                                "label": entity,
                                "label_id": entity
                            }
                            paragraph = meta["left_context"] \
                                + " [START] " \
                                + meta["mention"] \
                                  + " [END] " \
                                  + meta["right_context"]
                            sentences.append(paragraph)

                            qid, text = model.predict_paragraph(
                                paragraph, split_sentences=False, split_long_texts=False)[0]
                            for pos in entity[-1]:
                                pos_qid[pos] = qid
                                pos_ner[pos] = 'I-' + entity[1]

                            pos_ner[entity[-1][0]] = 'B-' + entity[1]
                        for idx, token in enumerate(tokens):

                            f.write(token + '\t')
                            if idx in pos_ner:
                                f.write(
                                    pos_ner[idx] +
                                    '\tO\tO\tO\tO\tO\t' +
                                    pos_qid[idx] +
                                    '\tO\tO\t' +
                                    text +
                                    '\n')
                            else:
                                f.write('O\tO\tO\tO\tO\tO\tO\tO\tO\n')

                        f.write('\n')

                # results = model.sample(
                #     sentences,
                #     prefix_allowed_tokens_fn=lambda batch_id, sent: [
                #         e for e in trie.get(sent.tolist())
                #         if e < len(model.task.target_dictionary)
                #         # for huggingface/transformers
                #         # if e < len(model2.tokenizer) - 1
                #     ],
                #     text_to_id=lambda x: max(lang_title2wikidataID[tuple(reversed(x.split(" >> ")))],
                #                              key=lambda y: int(y[1:])),
                #     marginalize=True,
                # )
                # for result, phrase in zip(results, phrase):
                #     print(result)

                        # wikidataIDs = get_wikidata_ids(
                        #     #title.replace("_", " "),
                        #     meta['label_title'],
                        #     lang,
                        #     lang_title2wikidataID,
                        #     lang_redirect2title,
                        #     label_or_alias2wikidataID,
                        # )[0]
                        #
                        # is_hard = False
                        # item = {
                        #     "id": "HIPE-{}-{}-{}".format(lang, filename, i),
                        #     "input": (
                        #         meta["left_context"]
                        #         + " [START] "
                        #         + meta["mention"]
                        #         + " [END] "
                        #         + meta["right_context"]
                        #     ),
                        #     "output": [{"answer": list(wikidataIDs)}],
                        #     "meta": meta,
                        #     "is_hard": is_hard,
                        # }
                        # kilt_dataset.append(item)
                        # import pdb;
                        #
                        # pdb.set_trace()

                    # for i, mention in enumerate(mentions):
                    #     start, end, _, title, is_hard = mention.strip().split("\t")
                    #     start, end, is_hard = int(start), int(end), bool(int(is_hard))
                    #     wikidataIDs = get_wikidata_ids(
                    #         title.replace("_", " "),
                    #         lang,
                    #         lang_title2wikidataID,
                    #         lang_redirect2title,
                    #         label_or_alias2wikidataID,
                    #     )[0]
                    #
                    #     meta = {
                    #         "left_context": doc[:start].strip(),
                    #         "mention": doc[start:end].strip(),
                    #         "right_context": doc[end:].strip(),
                    #     }
                    #     item = {
                    #         "id": "TR2016-{}-{}-{}".format(lang, filename, i),
                    #         "input": (
                    #             meta["left_context"]
                    #             + " [START] "
                    #             + meta["mention"]
                    #             + " [END] "
                    #             + meta["right_context"]
                    #         ),
                    #         "output": [{"answer": list(wikidataIDs)}],
                    #         "meta": meta,
                    #         "is_hard": is_hard,
                    #     }
                    #     kilt_dataset.append(item)
                    #
                    # filename = os.path.join(
                    #     args.output_dir, lang, "{}-kilt-{}.jsonl".format(lang, split)
                    # )
                    # logging.info("Saving {}".format(filename))
                    # with jsonlines.open(filename, "w") as f:
                    #     f.write_all(kilt_dataset)
                    #
                    # kilt_dataset = [e for e in kilt_dataset if e["is_hard"]]
                    #
                    # filename = os.path.join(
                    #     args.output_dir, lang, "{}-hard.jsonl".format(filename.split(".")[0])
                    # )
                    # logging.info("Saving {}".format(filename))
                    # with jsonlines.open(filename, "w") as f:
                    #     f.write_all(kilt_dataset)
