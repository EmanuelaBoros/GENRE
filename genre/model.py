
from typing import List, Optional
from fairseq_model import mGENRE
from helper_pickle import pickle_load
from trie import Trie, MarisaTrie


class Model:
    def __init__(self,
                 # yago: bool,
                 model_name,
                 mention_trie,
                 lang_title2wikidataID):
        # if yago:
        #     model_name = "models/fairseq_e2e_entity_linking_aidayago"
        # else:
        #     model_name = "models/fairseq_e2e_entity_linking_wiki_abs"
        print('Loading {}'.format(model_name))
        self.model = mGENRE.from_pretrained(model_name).eval()
        if torch.cuda.is_available():
            print("move model to GPU...")
            self.model = self.model.cuda()
        self.mention_trie = Trie.load_from_dict(
            pickle_load(mention_trie, verbose=True))

        # self.mention_to_candidates_dict = pickle_load(mention_to_candidates_dict, verbose=True)
        # self.candidates_trie = pickle_load(candidates_trie, verbose=True)
        self.spacy_model = None
        self.lang_title2wikidataID = pickle_load(
            lang_title2wikidataID, verbose=True)

    def _ensure_spacy(self):
        if self.spacy_model is None:
            self.spacy_model = spacy.load("en_core_web_sm")

    def _split_sentences(self, text: str) -> List[str]:
        self._ensure_spacy()
        doc = self.spacy_model(text)
        sentences = [sent.text for sent in doc.sents]
        return sentences

    def _split_long_texts(self, text: str) -> List[str]:
        MAX_WORDS = 150
        split_parts = []
        sentences = self._split_sentences(text)
        part = ""
        n_words = 0
        for sentence in sentences:
            sent_words = len(sentence.split())
            if len(part) > 0 and n_words + sent_words > MAX_WORDS:
                split_parts.append(part)
                part = ""
                n_words = 0
            if len(part) > 0:
                part += " "
            part += sentence
            n_words += sent_words
        if len(part) > 0:
            split_parts.append(part)
        return split_parts

    def predict_paragraph(
            self,
            text: str,
            split_sentences: bool,
            split_long_texts: bool) -> str:
        if split_sentences:
            sentences = self._split_sentences(text)
        elif split_long_texts:
            sentences = self._split_long_texts(text)
        else:
            sentences = [text]
        predictions = []
        for sent in sentences:
            # print("IN:", sent)
            if len(sent.strip()) == 0:
                prediction = sent
                qid = 'NIL'
            else:
                qid, prediction = self.predict(sent)
            # print("PREDICTION:", prediction)
            predictions.append((qid, prediction))
        return predictions

    def predict_iteratively(self, text: str):
        text = self._preprocess(text)
        sentences = self._split_sentences(text)
        n_parts = 1
        while n_parts <= len(sentences):
            plural_s = "s" if n_parts > 1 else ""
            # print(f"INFO: Predicting {n_parts} part{plural_s}.")
            sents_per_part = len(sentences) / n_parts
            results = []
            did_fail = False
            for i in range(n_parts):
                start = int(sents_per_part * i)
                end = int(sents_per_part * (i + 1))
                part = " ".join(sentences[start:end])
                print("IN:", part)
                try:
                    result = self._query_model(part)[0]
                except Exception:
                    result = None
                # print("RESULT:", result)
                if result is not None and len(
                        result) > 0 and _is_prediction_complete(part, result[0]["text"]):
                    results.append(result[0]["text"])
                elif end - start == 1:
                    results.append(part)
                else:
                    did_fail = True
                    break
            if did_fail:
                n_parts += 1
            else:
                return " ".join(results)

    def _preprocess(self, text):
        text = text.replace("[", "")
        text = text.replace("]", "")
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        return text

    def _query_model(self, text):
        if len(text) > 0 and text[0] != " ":
            text = " " + text  # necessary to detect mentions in the beginning of a sentence
        sentences = [text]

        try:
            result = self.model.sample(
                sentences, prefix_allowed_tokens_fn=lambda batch_id, sent: [
                    e for e in self.mention_trie.get(sent.tolist())
                    if e < len(self.model.task.target_dictionary)
                    # for huggingface/transformers
                    # if e < len(model2.tokenizer) - 1
                ],
                text_to_id=lambda x: max(self.lang_title2wikidataID[tuple(
                    reversed(x.split(" >> ")))], key=lambda y: int(y[1:])),
                marginalize=True,
            )
        except Exception as e:
            print('Sentence too long:', e)
            print(sentences[0])
            result = [
                [{"texts": "SENTENCE TOO LONG", "id": "NIL", "score": 0.0}]]

        return result

    def predict(self, text: str) -> str:
        # text = self._preprocess(text)

        result = self._query_model(text)

        text = result[0][0]["texts"][0].strip()
        qid = result[0][0]["id"]
        score = result[0][0]['score'].item()

        # is score ==
        return qid, text


def _is_prediction_complete(text, prediction):
    len_text = 0
    for char in text:
        if char != " ":
            len_text += 1
    len_prediction = 0
    inside_prediction = False
    for char in prediction:
        if char in " {}":
            continue
        elif char == "[":
            inside_prediction = True
        elif char == "]":
            inside_prediction = False
        elif not inside_prediction:
            len_prediction += 1
    return len_text == len_prediction
