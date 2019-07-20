import torch
from src.lib.inference_utils import create_drop_reader, load_model, data_instance_to_model_input


def extract_instance_properties(instance):
    metadata = instance['metadata'].metadata

    question_id = metadata['question_id']
    passage_id = metadata['passage_id']

    question_tokens = metadata['question_tokens']
    passage_tokens = metadata['passage_tokens']

    passage_numbers = [passage_tokens[idx] for idx in metadata['number_indices']]

    answer_texts = instance['metadata'].metadata['answer_texts']

    answer_as_passage_spans = metadata['answer_info']['answer_passage_spans']
    is_answer_psg_span = 0 if len(answer_as_passage_spans) == 0 else 1

    answer_as_question_spans = metadata['answer_info']['answer_question_spans']
    is_answer_qstn_span = 0 if len(answer_as_question_spans) == 0 else 1

    # TODO (Or Perel): TBD - need to decide what we do with that one
    signs_for_add_sub_expressions = metadata['answer_info']['signs_for_add_sub_expressions']

    counts = metadata['answer_info']['counts']
    is_answer_counts = 0 if len(counts) == 0 else 1

    is_answer_number = any([len(answer['number']) > 0 for answer in metadata['answer_annotations']])
    is_answer_date = any([any([val != "" for val in answer['date'].values()])
                          for answer in metadata['answer_annotations']])
    is_answer_span = any([len(answer['spans']) > 0 for answer in metadata['answer_annotations']])

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
        is_answer_counts=is_answer_counts,
        is_answer_number=is_answer_number,
        is_answer_date=is_answer_date
    )

    return entry

def featurize_entry(entry):

    return torch.tensor([
        entry['is_answer_span'],
        entry['is_answer_psg_span'],
        entry['is_answer_qstn_span'],
        entry['is_answer_counts'],
        entry['is_answer_number'],
        entry['is_answer_date']
    ])


model, config = load_model(model_path='results/naqanet_single_epoch/model.tar.gz',
                           weights_path='results/naqanet/best.th')
instances = create_drop_reader(config, data_split='dev', lazy=True)

feature_vecs = []

for instance_idx, instance in enumerate(instances):

    # Extract question / answer features here
    entry = extract_instance_properties(instance)
    feature_vec = featurize_entry(entry)

    model_input = data_instance_to_model_input(instance, model)
    prediction = model(**model_input)

    # TODO (Or Perel): Decide how to mark the prediction here..
    em_correct_label = 1 if prediction['answer'][0]['value'] == entry['answer_texts'][0] else 0

    feature_vecs.append((feature_vec, em_correct_label))


# TODO (Or Perel): Call Tensorboard summary object here with feature vecs..
