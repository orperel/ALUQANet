import string
from typing import List
from collections import Counter, defaultdict
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.dataset_readers.reading_comprehension.util import split_tokens_by_hyphen
from overrides import overrides
from word2number.w2n import word_to_num
from pytorch_pretrained_bert import BertTokenizer
from allennlp.pretrained import fine_grained_named_entity_recognition_with_elmo_peters_2018, \
    span_based_constituency_parsing_with_elmo_joshi_2018
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter


class BertDropTokenizer(Tokenizer):
    def __init__(self, pretrained_model: str):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(token) for token in self.tokenizer.tokenize(text)]


def _get_number_from_word(word):
    punctruations = string.punctuation.replace('-', '')
    word = word.strip(punctruations)
    word = word.replace(",", "")
    try:
        number = word_to_num(word)
    except ValueError:
        try:
            number = int(word)
        except ValueError:
            try:
                number = float(word)
            except ValueError:
                number = None
    return number


tokenizer = BertDropTokenizer(pretrained_model="bert-base-uncased")
number_tokenizer = WordTokenizer()
words_splitter = WordTokenizer()
sentences_splitter = SpacySentenceSplitter()
ner_tagger = fine_grained_named_entity_recognition_with_elmo_peters_2018()
pos_tagger = span_based_constituency_parsing_with_elmo_joshi_2018()


def extract_letters_frequency(passage, sentence_idx=None):
    """
    :param passage:
    :param sentence_idx: None for whole passage, else per sentence (index 0)..
    :return:
    """
    if sentence_idx is None:
        return dict(filter(lambda k: k[0].isalpha(), Counter(passage).items()))
    else:
        sentences = extract_sentences(passage)
        sen = sentences[sentence_idx]
        return dict(filter(lambda k: k[0].isalpha(), Counter(sen).items()))

def extract_words_frequency(passage, sentence_idx=None):
    """
    :param passage:
    :param sentence_idx: None for whole passage, else per sentence (index 0)..
    :return:
    """
    phrase = passage
    if sentence_idx is not None:
        sentences = extract_sentences(passage)
        phrase = sentences[sentence_idx]
    return Counter([str(tok) for tok in words_splitter.tokenize(phrase)])


def extract_words(passage):
    return words_splitter.tokenize(passage)

def extract_sentences(passage):
    return sentences_splitter.split_sentences(passage)

def extract_passage_numbers(passage):
    word_tokens = split_tokens_by_hyphen(number_tokenizer.tokenize(passage))
    numbers_in_passage = []
    number_indices = []
    number_words = []
    number_len = []
    passage_tokens = []
    curr_index = 0
    # Get all passage numbers
    for token in word_tokens:
        number = _get_number_from_word(token.text)
        wordpieces = tokenizer.tokenize(token.text)
        num_wordpieces = len(wordpieces)
        if number is not None:
            numbers_in_passage.append(number)
            number_indices.append(token.idx)
            number_words.append(token.text)
            number_len.append(num_wordpieces)
        passage_tokens += wordpieces
        curr_index += num_wordpieces

    return numbers_in_passage, number_indices

def extract_ner(passage, ner_to_extract=None):
    named_entities = defaultdict(list)
    sentences = extract_sentences(passage)
    for sentence in sentences:
        sentence_idx = passage.index(sentence)
        ner_tag_results = ner_tagger.predict(sentence=sentence)
        extracted_named_entities = [(word, ner_tag_results['tags'][word_idx])
                                    for word_idx, word in enumerate(ner_tag_results['words'])
                                    if ner_tag_results['tags'][word_idx] != 'O']

        remaining_sentence = sentence
        current_ne_tokens = []
        for word, named_entity in extracted_named_entities:
            named_entity_stripped = named_entity[2:]
            if named_entity.startswith('U-'):
                tagged_token_start_index = remaining_sentence.index(word) + sentence_idx
                tagged_token_end_index = tagged_token_start_index + len(word)
                named_entities[named_entity_stripped].append((word, tagged_token_start_index, tagged_token_end_index))
            elif named_entity.startswith('B-'):
                current_ne_tokens = []
                current_ne_tokens.append(word)
            elif named_entity.startswith('I-'):
                current_ne_tokens.append(word)
            elif named_entity.startswith('L-'):
                current_ne_tokens.append(word)
                ne = ' '.join(current_ne_tokens)
                tagged_token_start_index = remaining_sentence.index(ne) + sentence_idx
                tagged_token_end_index = tagged_token_start_index + len(ne)
                named_entities[named_entity_stripped].append((ne, tagged_token_start_index, tagged_token_end_index))

    if ner_to_extract is None:
        return named_entities
    else:
        retrieved_named_entities = []
        for category in ner_to_extract:
            retrieved_named_entities.extend(named_entities[category])
        return retrieved_named_entities


def extract_pos(passage, pos_to_extract=None):
    """
    :param passage:
    :param pos_to_extract: List
    :return:
    """
    spans = []
    sentences = extract_sentences(passage)
    for sentence in sentences:
        sentence_idx = passage.index(sentence)
        prediction = pos_tagger.predict(sentence)

        relevant_pos_tags = [tok for tok, pos in zip(prediction['tokens'], prediction['pos_tags'])
                             if pos in pos_to_extract]
        remaining_sentence = sentence

        for tagged_token in relevant_pos_tags:
            tagged_token_start_index = remaining_sentence.index(tagged_token) + sentence_idx
            tagged_token_end_index = tagged_token_start_index + len(tagged_token)
            spans.append((tagged_token_start_index, tagged_token_end_index))

    selected_spans_text = [passage[s[0]:s[1]] for s in spans]
    return spans, selected_spans_text

