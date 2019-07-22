from datetime import datetime
from collections import namedtuple
import torch
from src.lib.inference_utils import create_drop_reader, load_model, data_instance_to_model_input
from torch.utils.tensorboard import SummaryWriter


def get_instance_answer_types(instance):
    metadata = instance['metadata'].metadata

    answer_types = []
    if len(metadata['answer_info']['answer_passage_spans']) > 0:
        answer_types.append("passage_span")

    if len(metadata['answer_info']['answer_question_spans']) > 0:
        answer_types.append("question_span")

    if len(metadata['answer_info']['signs_for_add_sub_expressions']) > 0:
        answer_types.append("arithmetic")

    if len(metadata['answer_info']['counts']) > 0:
        answer_types.append("count")

    return answer_types


def extract_instance_properties(instance):
    metadata = instance['metadata'].metadata

    question_id = metadata['question_id']
    passage_id = metadata['passage_id']

    question_tokens = metadata['question_tokens']
    passage_tokens = metadata['passage_tokens']

    passage_numbers = [passage_tokens[idx] for idx in metadata['number_indices']]

    answer_texts = instance['metadata'].metadata['answer_texts']

    answer_as_passage_spans = metadata['answer_info']['answer_passage_spans']
    is_answer_psg_span = "passage_span" in get_instance_answer_types(instance)

    answer_as_question_spans = metadata['answer_info']['answer_question_spans']
    is_answer_qstn_span = "question_span" in get_instance_answer_types(instance)

    # TODO (Or Perel): TBD - need to decide what we do with that one
    signs_for_add_sub_expressions = metadata['answer_info']['signs_for_add_sub_expressions']
    is_answer_arithmetic = "arithmetic" in get_instance_answer_types(instance)

    counts = metadata['answer_info']['counts']
    is_answer_counts = "count" in get_instance_answer_types(instance)

    is_answer_number = any([len(answer['number']) > 0 for answer in metadata['answer_annotations']])
    is_answer_date = any([any([val != "" for val in answer['date'].values()])
                          for answer in metadata['answer_annotations']])
    is_answer_span = any([len(answer['spans']) > 0 for answer in metadata['answer_annotations']])

    question_contains_or = any(token == "or" for token in metadata['question_tokens'])

    question_about_football = any(token in term for term in ["touchdowns", "yards", "fields", "quarterbacks", "points", "passes", "kicks", "goals"] for token in metadata['question_tokens'])

    question_contains_percent = any(token in "percents" for token in metadata['question_tokens'])

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

def featurize_entry(entry):

    return torch.tensor([
        entry['is_answer_psg_span'],
        entry['is_answer_qstn_span'],
        entry['is_answer_counts'],
        entry['is_answer_arithmetic'],
        entry['is_answer_date'],
        entry['question_contains_or'],
        entry['question_about_football'],
        entry['question_contains_percent']
    ])


model, config = load_model(model_path='../results/naqanet_single_epoch/model.tar.gz',
                           weights_path='../results/naqanet/best.th')
instances = create_drop_reader(config, data_split='dev', lazy=True)

feature_vecs = []
labels = []
InstanceLabels = namedtuple('InstanceLabels',
                            ['em_correct', 'f1_score', 'f1_correct', 'answer_types', 'answer_type_correct',
                             'question_tokens', 'answer_texts', 'passage_numbers', 'passage_tokens'])

for instance_idx, instance in enumerate(instances):

    # Extract question / answer features here
    entry = extract_instance_properties(instance)
    feature_vec = featurize_entry(entry)

    # Extract labels here
    model_input = data_instance_to_model_input(instance, model)
    prediction = model(**model_input)

    metric = model.get_metrics(reset=True)
    em_correct_label = int(metric['em'])

    f1_threshold = 0.8
    f1_score = metric['f1']
    f1_score_label = '%.2f' % f1_score
    f1_correct_label = 1 if f1_score > f1_threshold else 0

    answer_types = get_instance_answer_types(instance)
    answer_type_correct = 1 if prediction["answer"][0]["answer_type"] in answer_types else 0
    answer_type_text = ' '.join(answer_types) if len(answer_types) > 0 else "None"

    passage_numbers = ' '.join([str(num) for num in entry['passage_numbers']]) if len(entry['passage_numbers']) > 0 else "None"

    instance_labels = InstanceLabels(em_correct=em_correct_label,
                                     f1_score=f1_score_label,
                                     f1_correct=f1_correct_label,
                                     answer_types=answer_type_text,
                                     answer_type_correct=answer_type_correct,
                                     question_tokens=' '.join(entry['question_tokens']),
                                     answer_texts=' '.join(entry['answer_texts']),
                                     passage_numbers=passage_numbers,
                                     passage_tokens=' '.join(entry['passage_tokens']),
                                     )

    feature_vecs.append(feature_vec)
    labels.append(tuple(instance_labels))

# Writer will output to ./tb_data_analysis/ directory
# (1) To run the TB server use:
# tensorboard --logdir=tb_data_analysis
# (2) Forward the display to local machine with:
# ssh -i <PATH_TO_REMOTE_MACHINE_SSH_KEY> -N -L localhost:6006:localhost:6006 ubuntu@<REMOTE_IP>
writer = SummaryWriter("tb_data_analysis/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
feature_vecs_tensor = torch.stack(feature_vecs)
writer.add_embedding(feature_vecs_tensor, metadata=labels, tag='all_features', metadata_header=InstanceLabels._fields)
writer.close()


