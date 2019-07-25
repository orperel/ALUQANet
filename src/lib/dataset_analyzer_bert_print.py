from datetime import datetime
from collections import namedtuple
import json
import torch

from drop_bert.augmented_bert_plus import NumericallyAugmentedBERTPlus
from drop_bert.data_processing import BertDropTokenizer, BertDropTokenIndexer, BertDropReader
from src.lib.inference_utils import create_drop_reader, load_model, data_instance_to_model_input
from torch.utils.tensorboard import SummaryWriter

from allennlp.data.vocabulary import Vocabulary


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

    question_tokens = metadata['original_question'].split(' ')
    passage_tokens = metadata['original_passage'].split(' ')

    passage_numbers = metadata['original_numbers']

    answer_texts = instance['metadata'].metadata['answer_texts']

    is_answer_number = any([len(answer['number']) > 0 for answer in metadata['answer_annotations']])
    is_answer_date = any([any([val != "" for val in answer['date'].values()])
                          for answer in metadata['answer_annotations']])
    is_answer_span = any([len(answer['spans']) > 0 for answer in metadata['answer_annotations']])

    entry = dict(
        question_tokens=question_tokens,
        passage_tokens=passage_tokens,
        passage_numbers=passage_numbers,
        answer_texts=answer_texts,
        is_answer_span=is_answer_span,
        is_answer_number=is_answer_number,
        is_answer_date=is_answer_date,
    )

    return entry


device_num = 0
device = torch.device('cuda:%d' % device_num)

tokenizer = BertDropTokenizer('bert-base-uncased')
token_indexer = BertDropTokenIndexer('bert-base-uncased')
model = NumericallyAugmentedBERTPlus(Vocabulary(), 'bert-base-uncased', special_numbers=[100, 1])
reader = BertDropReader(tokenizer, {'tokens': token_indexer},
                        extra_numbers=[100, 1])
model_weights = '/home/itaysofer/Desktop/Drop/run/nabert/serialization2/best.th'

model.load_state_dict(torch.load(model_weights, map_location=device))
model.to(device)
model.eval()

reader.answer_type = None
instances = reader.read('../../data/drop_dataset/drop_dataset_dev.json')

feature_vecs = []
labels = []

examples = []
for instance_idx, instance in enumerate(instances):

    # Extract labels here
    model_input = data_instance_to_model_input(instance, model)
    prediction = model(**model_input)

    metric = model.get_metrics(reset=True)
    em_correct_label = int(metric['em'])
    if em_correct_label == 1:
        continue

    # Extract question / answer features here
    entry = extract_instance_properties(instance)

    answer_types = get_instance_answer_types(instance)
    answer_type_text = ' '.join(answer_types) if len(answer_types) > 0 else "None"
    passage_numbers = ' '.join([str(num) for num in entry['passage_numbers']]) if len(entry['passage_numbers']) > 0 else "None"

    answer_contents = []
    if entry['is_answer_number']:
        answer_contents.append("number")
    if entry['is_answer_span']:
        answer_contents.append("span")
    if entry['is_answer_date']:
        answer_contents.append("date")

    answer_content = ' '.join(answer_contents) if len(answer_contents) > 0 else "None"

    example = {"answer_types": answer_type_text,
               "answer_content": answer_content,
               "passage_tokens": ' '.join(entry['passage_tokens']),
               "passage_numbers": passage_numbers,
               "question_tokens": ' '.join(entry['question_tokens']),
               "golden_answer": ' '.join(entry['answer_texts']),
               "predicted_answer": prediction["answer"][0]["value"]
               }

    examples.append(example)

    # if len(examples) > 10:
    #     break

span_examples = list(filter(lambda e: "span" in e["answer_content"], examples))
number_examples = list(filter(lambda e: "number" in e["answer_content"], examples))
date_examples = list(filter(lambda e: "date" in e["answer_content"], examples))


with open("./examples/span_examples.json","w+") as f:
    json.dump(span_examples, f)

with open("./examples/number_examples.json","w+") as f:
    json.dump(number_examples, f)

with open("./examples/date_examples.json", "w+") as f:
    json.dump(date_examples, f)
