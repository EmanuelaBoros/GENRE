import argparse
import json
import logging
import os
import random
import time
import torch
from datetime import timedelta
from nltk import pos_tag
from nltk.tree import Tree
import re

# Loading WIKIDATA client for getting EL
from wikidata.client import Client
client = Client()  # doctest: +SKIP


def _read_conll(path, encoding='utf-8', sep=None, indexes=None, dropna=True):
    r"""
    Construct a generator to read conll items.
    :param path: file path
    :param encoding: file's encoding, default: utf-8
    :param sep: seperator
    :param indexes: conll object's column indexes that needed, if None, all columns are needed. default: None
    :param dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True
    :return: generator, every time yield (line number, conll item)
    """

    def parse_conll(sample):

        sample = list(map(list, zip(*sample)))
        sample = [sample[i] for i in indexes]

        for f in sample:
            if len(f) <= 0:
                raise ValueError('empty field')
        return sample

    with open(path, 'r', encoding=encoding) as f:

        sample = []
        start = next(f).strip() # Skip columns
        start = next(f).strip()

        data = []
        for line_idx, line in enumerate(f, 0):
            line = line.strip()

            if any(substring in line for substring in ['DOCSTART', '###', "# id", "# ", '###']):
                continue

            if line == '':
                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        if ['TOKEN'] not in res:
                            if ['Token'] not in res:
                                data.append([line_idx, res])
                    except Exception as e:
                        if dropna:
                            print(
                                'Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            sample = []
                            raise e
                        raise ValueError(
                            'Invalid instance which ends at line: {}'.format(line_idx))
            elif 'EndOfSentence' in line:
                sample.append(
                    line.split(sep)) if sep else sample.append(
                    line.split())

                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        if ['TOKEN'] not in res:
                            if ['Token'] not in res:
                                data.append([line_idx, res])
                    except Exception as e:
                        if dropna:
                            print(
                                'Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            sample = []
                            raise e
                        raise ValueError(
                            'Invalid instance which ends at line: {}'.format(line_idx))
            else:
                sample.append(
                    line.split(sep)) if sep else sample.append(
                    line.split())

        if len(sample) > 0:
            try:
                res = parse_conll(sample)
                if ['TOKEN'] not in res:
                    if ['Token'] not in res:
                        data.append([line_idx, res])
            except Exception as e:
                if dropna:
                    return
                print('Invalid instance ends at line: {}'.format(line_idx))
                raise e

        return data


class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def conlltags2tree(sentence, chunk_types=('NP','PP','VP'),
                   root_label='S', strict=False):
    """
    Convert the CoNLL IOB format to a tree.
    """
    tree = Tree(root_label, [])
    for (word, chunktag, idx) in sentence:
        idx = str(idx)
        if chunktag is None:
            if strict:
                raise ValueError("Bad conll tag sequence")
            else:
                # Treat as O
                tree.append((word, idx))
        elif chunktag.startswith('B-'):
            tree.append(Tree(chunktag[2:], [(word, idx)]))
        elif chunktag.startswith('I-'):
            if len(tree)==0 or not isinstance(tree[-1], Tree) or tree[-1].label() != chunktag[2:]:
                if strict:
                    raise ValueError("Bad conll tag sequence")
                else:
                    # Treat as B-*
                    tree.append(Tree(chunktag[2:], [(word, idx)]))
            else:
                tree[-1].append((word, idx))
        elif chunktag == 'O':
            tree.append((word, idx))
        else:
            raise ValueError("Bad conll tag %r" % chunktag)
    return tree


def get_entities(tokens, entity_tags, link_tags):
    # print(tokens, entity_tags, link_tags)
    entity_tags = [tag.replace('S-', 'B-').replace('E-', 'I-') for tag in entity_tags]
    link_tags = [entity_tags[idx][:2] + tag if tag !='_' else 'O' for idx, tag in enumerate(link_tags)]
    pos_tags = [pos for token, pos in pos_tag(tokens)]

    indices = list(range(len(entity_tags)))
    entity_tags = [(token, tag, idx) for token, pos, tag, idx in zip(tokens, pos_tags, entity_tags, indices)]
    link_tags = [(token, tag, idx) for token, pos, tag, idx in zip(tokens, pos_tags, link_tags, indices)]

    ne_tree = conlltags2tree(entity_tags)
    el_tree = conlltags2tree(link_tags)

    entities = []
    for ne_subtree, el_subtree in zip(ne_tree, el_tree):
        # skipping 'O' tags
        # print(ne_subtree, '---', el_subtree)
        if type(ne_subtree) == Tree:
            en_label = ne_subtree.label()
            if type(el_subtree) != str and type(el_subtree) != tuple:
                el_label = el_subtree.label()
            else:
                el_label = 'NIL'
            en_string = " ".join([token[0] for token in ne_subtree.leaves()])
            en_idx = [int(token[1]) for token in ne_subtree.leaves()]
            entities.append((en_string, en_label, el_label, en_idx))
    return entities

"""
{
  "context_left": "3 console.\nSome users claim that Nvidia's Linux drivers impose artificial restrictions, 
  like limiting the number of monitors that can be used at the same time, but the company has not commented on these 
  accusations.\nIn 2014, with Maxwell GPUs, Nvidia started to require firmware by them to unlock all features of its 
  graphics cards. Up to now, this state has not changed and makes writing open-source drivers difficult.\nDeep learning.
  \nNvidia GPUs are often used in deep learning, and accelerated analytics due to Nvidia's API CUDA which allows",
  "context_right": "announced at Tesla Autonomy Day in 2019 that the company developed its own SoC and Full Self-Driving computer now and would stop using Nvidia hardware for their vehicles. According to \"TechRepublic\", Nvidia GPUs \"work well for deep learning tasks because they are designed for parallel computing and do well to handle the vector and matrix operations that are prevalent in deep learning\". These GPUs are used by researchers, laboratories, tech companies and enterprise companies. In 2009, Nvidia was involved in what was called the \"big bang\" of deep learning, \"as deep-learning neural networks were combined with Nvidia graphics processing units (GPUs)\". That year, the Google Brain used Nvidia GPUs to create Deep Neural Networks capable of machine learning, where Andrew Ng determined that GPUs could increase the speed",
  "mention": "Elon Musk",
  "label_title": "Elon Musk",
  "label": "Elon Reeve Musk (; born June 28, 1971) is an entrepreneur and business magnate. He is the founder, CEO and chief engineer at SpaceX; early stage investor, CEO, and product architect of Tesla, Inc.; founder of The Boring Company; and co-founder of Neuralink and OpenAI. A centibillionaire, Musk is one of the richest people in the world.\nMusk was born to a Canadian mother and South African father and raised in Pretoria, South Africa. He briefly attended the University of Pretoria before moving to Canada aged 17 to attend Queen's University. He transferred to the University of Pennsylvania two years later, where he received bachelors' degrees in economics and physics. He moved to California in 1995 to attend Stanford University but decided instead to pursue a business career, co-founding",
  "label_id": 304371
}
"""

def convert_data_(params, entity_dict, entity_map, filename):

    fout = open(os.path.join(params.output_path, filename.replace('tsv', ".jsonl")), 'wt')
    cnt = 0
    max_tok = 128
    with open(filename, 'rt') as f:
        for line in f:
            cnt += 1
            line = line.rstrip()
            item = json.loads(line)
            mention = item["text"].lower()
            src = item["corpus"]
            label_doc_id = item["label_document_id"]
            orig_doc_id = item["context_document_id"]
            start = item["start_index"]
            end = item["end_index"]

            # add context around the mention as well
            orig_id = entity_map[src][orig_doc_id]
            text = entity_dict[src][orig_id]["text"].lower()
            tokens = text.split(" ")

            assert mention == ' '.join(tokens[start:end + 1])
            tokenized_query = mention

            mention_context_left = tokens[max(0, start - max_tok):start]
            mention_context_right = tokens[end + 1:min(len(tokens), end + max_tok + 1)]

            # entity info
            k = entity_map[src][label_doc_id]
            ent_title = entity_dict[src][k]['title']
            ent_text = entity_dict[src][k]["text"]

            example = {}
            example["context_left"] = ' '.join(mention_context_left)
            example['context_right'] = ' '.join(mention_context_right)
            example["mention"] = mention
            example["label"] = ent_text
            example["label_id"] = k
            example['label_title'] = ent_title
            example['world'] = src
            fout.write(json.dumps(example))
            fout.write('\n')

    fout.close()

def convert_data(params, entity_dict, entity_map, filename):

    fout = open(os.path.join(params.output_path, filename.replace('tsv', ".jsonl")), 'wt')
    cnt = 0
    max_tok = 128
    with open(filename, 'rt') as f:
        for line in f:
            cnt += 1
            line = line.rstrip()
            item = json.loads(line)
            mention = item["text"].lower()
            src = item["corpus"]
            label_doc_id = item["label_document_id"]
            orig_doc_id = item["context_document_id"]
            start = item["start_index"]
            end = item["end_index"]

            # add context around the mention as well
            orig_id = entity_map[src][orig_doc_id]
            text = entity_dict[src][orig_id]["text"].lower()
            tokens = text.split(" ")

            assert mention == ' '.join(tokens[start:end + 1])
            tokenized_query = mention

            mention_context_left = tokens[max(0, start - max_tok):start]
            mention_context_right = tokens[end + 1:min(len(tokens), end + max_tok + 1)]

            # entity info
            k = entity_map[src][label_doc_id]
            ent_title = entity_dict[src][k]['title']
            ent_text = entity_dict[src][k]["text"]

            example = {}
            example["context_left"] = ' '.join(mention_context_left)
            example['context_right'] = ' '.join(mention_context_right)
            example["mention"] = mention
            example["label"] = ent_text
            example["label_id"] = k
            example['label_title'] = ent_title
            example['world'] = src
            fout.write(json.dumps(example))
            fout.write('\n')

    fout.close()


if __name__ == '__main__':

    log_formatter = LogFormatter()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser(description='Zero-shot Entity Linking Dataset')
    parser.add_argument(
        '--documents_path',
        default='data/documents', # Wikipedia?
        type=str,
    )
    parser.add_argument(
        '--mentions_path',
        default='data/mentions', #HIPE predictions or gold truth
        type=str,
    )
    parser.add_argument(
        '--output_path',
        default='data/blink_format',
        type=str,
    )
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    for root, dirs, files in os.walk(args.mentions_path, topdown=False):
        for filename in files:
            if 'tsv' in filename:
                filename = os.path.join(root, filename)
                print(filename)
                with open(filename, 'r') as f:
                    lines = f.readlines()

                headers = [
                    'raw_words', 'target', 'link'
                ]
                # TODO: This needs to be changed if the data format is different or the
                # order of the elements in the file is different
                indexes = [0, 1, -3] # -3 is for EL
                if not isinstance(headers, (list, tuple)):
                    raise TypeError(
                        'invalid headers: {}, should be list of strings'.format(headers))
                phrases = _read_conll(filename, encoding='utf-8', sep='\t', indexes=indexes, dropna=True)

                entity = client.get('Q7344037', load=True)

                for phrase in phrases:
                    # [('R . Ellis', 'pers', 'Q7344037'), ('the Cambridge Journal of Philology', 'work', 'NIL'),
                    # ('Vol . IV', 'scope', 'NIL'), ('A . Nauck', 'pers', 'NIL'), ('Leipzig', 'loc', 'NIL'), ('1856',
                    # 'date', 'NIL')]

                    _, phrase = phrase
                    tokens, entity_tags, link_tags = phrase
                    entities = get_entities(tokens, entity_tags, link_tags)

                    # [('R . Ellis', 'pers', 'Q7344037', [12, 13, 14])
                    for entity in entities:
                        json_entity = {
                            "context_left": ' '.join(tokens[:entity[-1][0]]),
                            "context_right": ' '.join(tokens[entity[-1][-1]:]),
                            "mention": entity[0],
                            "label_title": entity,
                            "label": entity,
                            "label_id": entity
                        }
                    import pdb;
                    pdb.set_trace()

                # out_file = args.in_file.replace(".tsv", ".txt")
                # with open(out_file, "w") as f:
                #     f.writelines(phrases)
    # {
    #     "context_left": "3 console.\nSome users claim that Nvidia's Linux drivers impose artificial restrictions,
    #     like limiting the number of monitors that can be used at the same time, but the company has not commented on
    #     these
    #         accusations.\nIn
    # 2014,
    # with Maxwell GPUs, Nvidia started to require firmware by them to unlock all features of its
    # graphics cards.Up to now, this state has not changed and makes writing open-source drivers difficult.\nDeep learning.
    # \nNvidia GPUs are often used in deep learning, and accelerated analytics due to Nvidia's API CUDA which allows",
    # "context_right": "announced at Tesla Autonomy Day in 2019 that the company developed its own SoC and Full Self-Driving computer now and would stop using Nvidia hardware for their vehicles. According to \"TechRepublic\", Nvidia GPUs \"work well for deep learning tasks because they are designed for parallel computing and do well to handle the vector and matrix operations that are prevalent in deep learning\". These GPUs are used by researchers, laboratories, tech companies and enterprise companies. In 2009, Nvidia was involved in what was called the \"big bang\" of deep learning, \"as deep-learning neural networks were combined with Nvidia graphics processing units (GPUs)\". That year, the Google Brain used Nvidia GPUs to create Deep Neural Networks capable of machine learning, where Andrew Ng determined that GPUs could increase the speed",
    # "mention": "Elon Musk",
    # "label_title": "Elon Musk",
    # "label": "Elon Reeve Musk (; born June 28, 1971) is an entrepreneur and business magnate. He is the founder, CEO and chief engineer at SpaceX; early stage investor, CEO, and product architect of Tesla, Inc.; founder of The Boring Company; and co-founder of Neuralink and OpenAI. A centibillionaire, Musk is one of the richest people in the world.\nMusk was born to a Canadian mother and South African father and raised in Pretoria, South Africa. He briefly attended the University of Pretoria before moving to Canada aged 17 to attend Queen's University. He transferred to the University of Pennsylvania two years later, where he received bachelors' degrees in economics and physics. He moved to California in 1995 to attend Stanford University but decided instead to pursue a business career, co-founding",
    # "label_id": 304371
    # }
    # entity_dict, entity_map = load_entity_dict(params)
    # convert_data(params, entity_dict, entity_map, 'train')
    # convert_data(params, entity_dict, entity_map, 'valid')
    # convert_data(params, entity_dict, entity_map, 'test')
