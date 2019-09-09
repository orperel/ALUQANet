import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../src/lib")
sys.path.append("../../src/lib")

import torch
import json
from drop_bert.data_processing import BertDropTokenizer, BertDropTokenIndexer, BertDropReader
from aluqa_or.generators.curicculum.class_a import CuricculumClassA
from aluqa_or.generators.curicculum.class_b import CuricculumClassB
from aluqa_or.generators.curicculum.class_c import CuricculumClassC
from aluqa_or.generators.curicculum.class_d import CuricculumClassD
from aluqa_or.generators.curicculum.class_e import CuricculumClassE
from aluqa_or.generators.curicculum.class_f import CuricculumClassF


classA = CuricculumClassA()
classB = CuricculumClassB()
classC = CuricculumClassC()
classD = CuricculumClassD()
classE = CuricculumClassE()
classF = CuricculumClassF()


def create_nabert_reader(data_path, lazy=True):
    tokenizer = BertDropTokenizer('bert-base-uncased')
    token_indexer = BertDropTokenIndexer('bert-base-uncased')
    reader = BertDropReader(tokenizer, {'tokens': token_indexer},
                            extra_numbers=[100, 1], lazy=lazy)
    reader.answer_type = None
    instances = reader.read(data_path)

    return instances

def dump_results(data_to_dump, filename):
    with open(filename, 'w') as f:
        json.dump(data_to_dump, f)

def _generate_question_for_class(passage, class_generator, class_symbol):
    samples = []
    amount_per_class = 10
    for _ in range(amount_per_class):
        try:
            sample = class_generator.sample(passage)
            sample['question_class'] = class_symbol
            samples.append(sample)
        except:
            continue
    return samples

def generate_questions(passage, query_id):
    samples = []
    samples.extend(_generate_question_for_class(passage, classA, 'A'))
    samples.extend(_generate_question_for_class(passage, classB, 'B'))
    samples.extend(_generate_question_for_class(passage, classC, 'C'))
    samples.extend(_generate_question_for_class(passage, classD, 'D'))
    samples.extend(_generate_question_for_class(passage, classE, 'E'))
    samples.extend(_generate_question_for_class(passage, classF, 'F'))

    converted_qa_pairs = [
        dict(
            question=entry['question'],
            question_class=entry['question_class'],
            span_type=entry['span_type'],
            char_span_indices=entry['spans'],
            answer={
                "number": str(entry['answer']),
                "date": {
                    "day": "",
                    "month": "",
                    "year": ""
                },
                "spans": []
            },
            query_id=f"{idx+query_id}"
        ) for idx, entry in enumerate(samples)
    ]
    converted_samples = {
            'passage': passage,
            'qa_pairs': converted_qa_pairs,
            'wiki_url': 'None'
    }
    query_id += len(converted_qa_pairs)
    return converted_samples, query_id

split = 'train'
instances = create_nabert_reader(data_path=f'../../../data/drop_dataset/drop_dataset_{split}.json')
generated_data = dict()
query_id = 0
with torch.no_grad():
    for instance_idx, instance in enumerate(instances):
        passage = instance.fields['metadata'].metadata['original_passage']
        passage_id = instance.fields['metadata'].metadata['passage_id']

        new_samples, query_id = generate_questions(passage, query_id)
        generated_data[passage_id] = new_samples

        if instance_idx % 201 == 200:
            dump_results(generated_data, f'curicculum_{split}.json')

dump_results(generated_data, f'curicculum_{split}.json')

print('Done.')
