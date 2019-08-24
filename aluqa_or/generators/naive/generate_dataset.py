import json
import random

import numpy as np
from aluqa_or.generators.naive.naive_generator import NaiveQAGenerator

num_instances = 10
train_percent = 0.9

np.random.seed(123)
random.seed(123)
qa_generator = NaiveQAGenerator()
instances = dict()

for idx in range(num_instances):
    question, answer, passage, \
    passage_span_indices, passage_spans, question_metadata, passage_metadata = qa_generator.generate()

    instance = dict()
    instance["passage"] = passage

    qa_pair = dict()
    qa_pair["question"] = question
    qa_pair["answer"] = {"number": str(answer),
                         "date": {"day": "", "month": "", "year": ""},
                         "spans": []}
    qa_pair["span_indices"] = passage_span_indices
    qa_pair["span_text"] = passage_spans
    qa_pair["question_metadata"] = question_metadata
    qa_pair["passage_metadata"] = passage_metadata
    qa_pair["query_id"] = ""

    instance["qa_pairs"] = [qa_pair]

    instances["Sample_" + str(idx)] = instance


num_train_instances = int(num_instances * train_percent)
train_instances = dict(list(instances.items())[:num_train_instances])
dev_instances = dict(list(instances.items())[num_train_instances:])


with open("generated_train.json", 'w+') as output_file:
    json.dump(train_instances, output_file)

with open("generated_dev.json", 'w+') as output_file:
    json.dump(dev_instances, output_file)
