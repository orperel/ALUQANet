import sys
import re
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../src/lib")
sys.path.append("../../src/lib")

from collections import defaultdict, Counter
import torch
import json
from drop_bert.data_processing import BertDropTokenizer, BertDropTokenIndexer, BertDropReader
from allennlp.pretrained import fine_grained_named_entity_recognition_with_elmo_peters_2018
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter


def create_nabert_reader(data_path, lazy=True):
    tokenizer = BertDropTokenizer('bert-base-uncased')
    token_indexer = BertDropTokenIndexer('bert-base-uncased')
    reader = BertDropReader(tokenizer, {'tokens': token_indexer},
                            extra_numbers=[100, 1], lazy=lazy)
    reader.answer_type = None
    instances = reader.read(data_path)

    return instances


def dump_aggregated_sentences(sentences, filename):
    with open(filename, 'w') as f:
        json.dump(sentences, f)

def mask_sentence(ner_tagger, sentence):
    ner_tag_results = ner_tagger.predict(sentence)
    masked_sentence = ''
    for word_idx, (word, tag) in enumerate(zip(ner_tag_results['words'], ner_tag_results['tags'])):
        if tag.startswith('O'):
            if word.endswith('-yard'):
                masked_sentence += ' [GOAL]'
            elif word_idx > 0 and word.istitle():    # Possible missed named entity, abort
                return None
            elif word.isnumeric():  # Possible missed cardinal, abort (floating points will prevail)
                return None
            elif re.search('[a-zA-Z0-9]', word) and '\'' not in word:
                masked_sentence += ' ' + word
            else:
                masked_sentence += word  # Non alpha-numeric token, i.e: , or .
        elif tag.startswith('U') or tag.startswith('B'):
            tag_mask = '[' + tag[2:] + ']'
            masked_sentence += ' ' + tag_mask
        else:
            continue
    return masked_sentence

instances = create_nabert_reader(data_path='../../data/drop_dataset/drop_dataset_train.json')
sentences_splitter = SpacySentenceSplitter()
ner_tagger = fine_grained_named_entity_recognition_with_elmo_peters_2018()

content_sentences = dict()

with torch.no_grad():
    for instance_idx, instance in enumerate(instances):
        collected_sentences_for_passage = defaultdict(list)
        passage_id = instance.fields['metadata'].metadata['passage_id']
        if passage_id in content_sentences:
            continue
        original_question = instance.fields['metadata'].metadata['original_question']
        original_passage = instance.fields['metadata'].metadata['original_passage']

        if any([w in original_passage for w in ('game', 'touchdown', 'match', 'player', 'yard', 'goal')]):
            passage_sentences = sentences_splitter.split_sentences(original_passage)

            for sentence_idx, sentence in enumerate(passage_sentences):
                masked_sentence = mask_sentence(ner_tagger, sentence)
                if masked_sentence is not None and len(masked_sentence) > 0:
                    collected_sentences_for_passage[sentence_idx].append(masked_sentence)

        content_sentences[passage_id] = collected_sentences_for_passage

        if len(content_sentences) % 201 == 200:
            dump_aggregated_sentences(content_sentences, 'content_sentences.json')

dump_aggregated_sentences(content_sentences, 'content_sentences.json')
print('Done.')
