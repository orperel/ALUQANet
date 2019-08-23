import sys
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
from allennlp.pretrained import named_entity_recognition_with_elmo_peters_2018
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter


def create_nabert_reader(data_path, lazy=True):
    tokenizer = BertDropTokenizer('bert-base-uncased')
    token_indexer = BertDropTokenIndexer('bert-base-uncased')
    reader = BertDropReader(tokenizer, {'tokens': token_indexer},
                            extra_numbers=[100, 1], lazy=lazy)
    reader.answer_type = None
    instances = reader.read(data_path)

    return instances


def aggregate_named_entities(sentence, named_entities):
    ner_tag_results = ner_tagger.predict(sentence=sentence)
    extracted_named_entities = [(word, ner_tag_results['tags'][word_idx])
                                for word_idx, word in enumerate(ner_tag_results['words'])
                                if ner_tag_results['tags'][word_idx] != 'O']

    current_ne_tokens = []
    for word, named_entity in extracted_named_entities:
        named_entity_stripped = named_entity[2:]
        if named_entity.startswith('U-'):
            named_entities[named_entity_stripped].append(word)
        elif named_entity.startswith('B-'):
            current_ne_tokens = []
            current_ne_tokens.append(word)
        elif named_entity.startswith('I-'):
            current_ne_tokens.append(word)
        elif named_entity.startswith('L-'):
            current_ne_tokens.append(word)
            ne = ' '.join(current_ne_tokens)
            named_entities[named_entity_stripped].append(ne)

def dump_frequency_table(aggregated_named_entities, filename):
    named_entities_frequency_table = {}
    for named_entity_category, words in aggregated_named_entities.items():
        word_counts = Counter(words)
        total_sum = sum(word_counts.values())
        word_frequencies = [(word, float(frequency) / total_sum) for word, frequency in word_counts.items()]
        named_entities_frequency_table[named_entity_category] = word_frequencies

    with open(filename, 'w') as f:
        json.dump(named_entities_frequency_table, f)

instances = create_nabert_reader(data_path='../../data/drop_dataset/drop_dataset_train.json')
ner_tagger = fine_grained_named_entity_recognition_with_elmo_peters_2018()
sentences_splitter = SpacySentenceSplitter()
named_entities = defaultdict(list)

with torch.no_grad():
    for instance_idx, instance in enumerate(instances):
        original_question = instance.fields['metadata'].metadata['original_question']
        original_passage = instance.fields['metadata'].metadata['original_passage']

        aggregate_named_entities(original_question, named_entities)

        # NER tagger is more accurate when single sentences are fed as input
        passage_sentences = sentences_splitter.split_sentences(original_passage)
        for passage_sentence in passage_sentences:
            aggregate_named_entities(passage_sentence, named_entities)

        if instance_idx % 501 == 500:
            dump_frequency_table(named_entities, 'ner_frequencies_latest.json')

dump_frequency_table(named_entities, 'ner_frequencies.json')

print('Done.')
