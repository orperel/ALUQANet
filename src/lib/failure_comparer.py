from datetime import datetime
from collections import namedtuple
import torch
import string
import numpy as np
from sklearn.decomposition import PCA

from drop_bert.augmented_bert_plus import NumericallyAugmentedBERTPlus
from aluqa.itay.aluqa_count import AluQACount
from drop_bert.data_processing import BertDropTokenizer, BertDropTokenIndexer, BertDropReader
from src.lib.inference_utils import create_nabert_drop_reader, load_model, data_instance_to_model_input
from torch.utils.tensorboard import SummaryWriter

from allennlp.data.vocabulary import Vocabulary


def load_aluqa_model(weights_path):
    device_num = 0
    device = torch.device('cuda:%d' % device_num)
    model = AluQACount(Vocabulary(), 'bert-base-uncased', special_numbers=[100, 1], answering_abilities=["counting"])
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model

def get_instance_answer_types(instance):
    metadata = instance['metadata'].metadata

    answer_types = []
    if len(metadata['answer_info']['answer_passage_spans']) > 0:
        answer_types.append("passage_span")

    if len(metadata['answer_info']['answer_question_spans']) > 0:
        answer_types.append("question_span")

    if len(metadata['answer_info']['expressions']) > 0:
        answer_types.append("arithmetic")

    if len(metadata['answer_info']['counts']) > 0:
        answer_types.append("count")

    return answer_types


def extract_instance_properties(instance):
    metadata = instance['metadata'].metadata

    question_id = metadata['question_id']
    passage_id = metadata['passage_id']

    question_tokens = metadata['original_question'].split(' ')
    passage_tokens = metadata['original_passage'].split(' ')

    passage_numbers = metadata['original_numbers']

    answer_texts = instance['metadata'].metadata['answer_texts']

    answer_as_passage_spans = metadata['answer_info']['answer_passage_spans']
    is_answer_psg_span = "passage_span" in get_instance_answer_types(instance)

    answer_as_question_spans = metadata['answer_info']['answer_question_spans']
    is_answer_qstn_span = "question_span" in get_instance_answer_types(instance)

    # TODO (Or Perel): TBD - need to decide what we do with that one
    signs_for_add_sub_expressions = metadata['answer_info']['expressions']
    is_answer_arithmetic = "arithmetic" in get_instance_answer_types(instance)

    counts = metadata['answer_info']['counts']
    is_answer_counts = "count" in get_instance_answer_types(instance)

    is_answer_number = any([len(answer['number']) > 0 for answer in metadata['answer_annotations']])
    is_answer_date = any([any([val != "" for val in answer['date'].values()])
                          for answer in metadata['answer_annotations']])
    is_answer_span = any([len(answer['spans']) > 0 for answer in metadata['answer_annotations']])

    question_contains_or = any(token == "or" for token in metadata['question_tokens'])

    question_about_football = any(token in term for term in ["touchdowns", "yards", "fields", "quarterbacks", "points", "passes", "kicks", "goals"] for token in question_tokens)

    question_contains_percent = any(token in "percents" for token in question_tokens)

    entry = dict(
        question_id=question_id,
        passage_id=passage_id,
        question_tokens=question_tokens,
        passage_tokens=passage_tokens,
        passage_numbers=passage_numbers,
        answer_texts=answer_texts,
        answer_as_passage_spans=answer_as_passage_spans,
        answer_as_question_spans=answer_as_question_spans,
        signs_for_add_sub_expressions=signs_for_add_sub_expressions,
        counts=counts,
        is_answer_span=is_answer_span,
        is_answer_psg_span=is_answer_psg_span,
        is_answer_qstn_span=is_answer_qstn_span,
        is_answer_arithmetic=is_answer_arithmetic,
        is_answer_counts=is_answer_counts,
        is_answer_number=is_answer_number,
        is_answer_date=is_answer_date,
        question_contains_or=question_contains_or,
        question_about_football=question_about_football,
        question_contains_percent=question_contains_percent
    )

    return entry


def join_tokens_to_readable_string(tokens):
    printable_symbols = set(string.printable)
    full_string = ' '.join(tokens)
    full_string = map(lambda x: x if x in printable_symbols else '*', full_string)
    full_string = ''.join(list(full_string))
    return full_string


nabertplus_model, nabertplus_config = load_model(model_path='/home/itaysofer/Desktop/Drop/run/nabertplus_count/serialization/model.tar.gz',
                           weights_path='/home/itaysofer/Desktop/Drop/run/nabertplus_count/serialization/best.th')

aluqa_model = load_aluqa_model('/home/itaysofer/Desktop/Drop/run/aluqa_count_entropy_fix/serialization/best.th')

instances = create_nabert_drop_reader(nabertplus_config, data_split='dev', lazy=True)


nabertplus_correct = []
aluqa_correct = []
all_question = []
for instance_idx, instance in enumerate(instances):

    entry = extract_instance_properties(instance)

    question_tokens = join_tokens_to_readable_string(entry['question_tokens'])
    passage_tokens = join_tokens_to_readable_string(entry['passage_tokens'])
    sample = {"question_tokens": question_tokens,
              "passage_tokens": passage_tokens}

    all_question.append(sample)

    # nabertplus
    model_input = data_instance_to_model_input(instance, nabertplus_model)
    prediction = nabertplus_model(**model_input)

    metric = nabertplus_model.get_metrics(reset=True)
    em_correct_label = int(metric['em'])
    if em_correct_label == 1:
        nabertplus_correct.append(sample)


    # aluqa
    model_input = data_instance_to_model_input(instance, aluqa_model)
    prediction = aluqa_model(**model_input)

    metric = aluqa_model.get_metrics(reset=True)
    em_correct_label = int(metric['em'])
    if em_correct_label == 1:
        aluqa_correct.append(sample)


nabertplus_correct_only = [item for item in nabertplus_correct if item not in aluqa_correct]
aluqa_correct_only = [item for item in aluqa_correct if item not in nabertplus_correct]
remained_failures = [item for item in all_question if item not in nabertplus_correct and item not in aluqa_correct]

print("")




