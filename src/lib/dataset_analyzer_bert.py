from datetime import datetime
from collections import namedtuple
import torch
import string
import numpy as np
from sklearn.decomposition import PCA

from drop_bert.augmented_bert_plus import NumericallyAugmentedBERTPlus
from drop_bert.data_processing import BertDropTokenizer, BertDropTokenIndexer, BertDropReader
from src.lib.inference_utils import data_instance_to_model_input
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


def featurize_entry(entry, question_vector, passage_vector, featureize_by='all_features'):

    with torch.no_grad():
        question_tokens_count = len(entry['question_tokens'])
        passage_tokens_count = len(entry['passage_tokens'])
        passage_numbers_count = len(entry['passage_numbers'])

        if featureize_by == 'all_features':
            return torch.tensor([
                entry['is_answer_number'],
                entry['is_answer_span'],
                entry['is_answer_date'],
                entry['is_answer_psg_span'],
                entry['is_answer_qstn_span'],
                entry['is_answer_counts'],
                entry['is_answer_arithmetic'],
                entry['question_contains_or'],
                entry['question_about_football'],
                entry['question_contains_percent'],
                question_tokens_count,
                passage_tokens_count,
                passage_numbers_count
            ])
        elif featureize_by == 'token_count':
            return torch.tensor([
                question_tokens_count,
                passage_tokens_count,
                passage_numbers_count
            ])
        elif featureize_by == 'question_vec':
            return question_vector.cpu()
        elif featureize_by == 'passage_vec':
            return passage_vector.cpu()
        elif featureize_by == 'question_passage_vec':
            return torch.cat((question_vector.cpu(), passage_vector.cpu()))
        elif featureize_by == 'all_features_qa_vec':
            all_features = torch.tensor([
                entry['is_answer_number'],
                entry['is_answer_span'],
                entry['is_answer_date'],
                entry['is_answer_psg_span'],
                entry['is_answer_qstn_span'],
                entry['is_answer_counts'],
                entry['is_answer_arithmetic'],
                entry['question_contains_or'],
                entry['question_about_football'],
                entry['question_contains_percent'],
                question_tokens_count,
                passage_tokens_count,
                passage_numbers_count
            ])
            return torch.cat((all_features.float(), question_vector.cpu(), passage_vector.cpu()))
        else:
            raise ValueError('Unknown featurize_by arg')


def load_nabert_model(weights_path):
    device_num = 0
    device = torch.device('cuda:%d' % device_num)
    model = NumericallyAugmentedBERTPlus(Vocabulary(), 'bert-base-uncased', special_numbers=[100, 1])
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def create_nabert_reader(data_path):
    tokenizer = BertDropTokenizer('bert-base-uncased')
    token_indexer = BertDropTokenIndexer('bert-base-uncased')
    reader = BertDropReader(tokenizer, {'tokens': token_indexer},
                            extra_numbers=[100, 1], lazy=True)
    reader.answer_type = None
    instances = reader.read(data_path)

    return instances


def pca(feature_vec):
    X = feature_vec.cpu().numpy()
    pca_func = PCA(n_components=25, svd_solver='auto')
    X_pca = pca_func.fit_transform(X)
    X_pca = torch.from_numpy(X_pca)
    return X_pca


model = load_nabert_model(weights_path='../results/nabert/best.th')
instances = create_nabert_reader(data_path='../../data/drop_dataset/drop_dataset_dev.json')


feature_vecs_all = []
feature_vecs_token_count = []
feature_question_vecs = []
feature_passage_vecs = []
feature_question_passage_vecs = []
feature_vecs_all_and_qa_vecs = []
labels = []
InstanceLabels = namedtuple('InstanceLabels',
                            ['em_correct', 'f1_score', 'f1_correct', 'answer_types', 'answer_type_correct',
                             'question_tokens', 'answer_texts', 'passage_numbers', 'passage_tokens', 'answer_content'])

for instance_idx, instance in enumerate(instances):

    entry = extract_instance_properties(instance)

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

    answer_contents = []
    if entry['is_answer_number']:
        answer_contents.append("number")
    if entry['is_answer_span']:
        answer_contents.append("span")
    if entry['is_answer_date']:
        answer_contents.append("date")

    answer_content = ' '.join(answer_contents) if len(answer_contents) > 0 else "None"

    question_tokens = join_tokens_to_readable_string(entry['question_tokens'])
    passage_tokens = join_tokens_to_readable_string(entry['passage_tokens'])
    answer_texts = join_tokens_to_readable_string(entry['answer_texts'])

    question_vector, passage_vector = model.extract_summary_vecs(**model_input)
    question_vector = question_vector.squeeze()
    passage_vector = passage_vector.squeeze()

    instance_labels = InstanceLabels(em_correct=em_correct_label,
                                     f1_score=f1_score_label,
                                     f1_correct=f1_correct_label,
                                     answer_types=answer_type_text,
                                     answer_type_correct=answer_type_correct,
                                     question_tokens=question_tokens,
                                     answer_texts=answer_texts,
                                     passage_numbers=passage_numbers,
                                     passage_tokens=passage_tokens,
                                     answer_content=answer_content
                                     )

    # Extract question / answer features here
    feature_vec_all = featurize_entry(entry, question_vector, passage_vector, featureize_by='all_features')
    feature_vec_token_count = featurize_entry(entry, question_vector, passage_vector, featureize_by='token_count')
    feature_question_vec = featurize_entry(entry, question_vector, passage_vector, featureize_by='question_vec')
    feature_passage_vec = featurize_entry(entry, question_vector, passage_vector, featureize_by='passage_vec')
    feature_question_passage_vec = featurize_entry(entry, question_vector, passage_vector, featureize_by='question_passage_vec')
    feature_all_features_qa_vec = featurize_entry(entry, question_vector, passage_vector, featureize_by='all_features_qa_vec')

    feature_vecs_all.append(feature_vec_all)
    feature_vecs_token_count.append(feature_vec_token_count)
    feature_question_vecs.append(feature_question_vec)
    feature_passage_vecs.append(feature_passage_vec)
    feature_question_passage_vecs.append(feature_question_passage_vec)
    feature_vecs_all_and_qa_vecs.append(feature_all_features_qa_vec)

    labels.append(tuple(instance_labels))


# Writer will output to ./tb_data_analysis/ directory
# (1) To run the TB server use:
# tensorboard --logdir=tb_data_analysis
# (2) Forward the display to local machine with:
# ssh -i <PATH_TO_REMOTE_MACHINE_SSH_KEY> -N -L localhost:6006:localhost:6006 ubuntu@<REMOTE_IP>
writer = SummaryWriter("tb_data_analysis/nabert_" + datetime.now().strftime("%Y%m%d-%H%M%S"))

feature_vec_options = dict(all_features=feature_vecs_all,
                           token_count_features=feature_vecs_token_count,
                           question_vec=feature_question_vecs,
                           passage_vec=feature_passage_vecs,
                           question_passage_vec=feature_question_passage_vecs,
                           all_features_qa_vec=feature_vecs_all_and_qa_vecs)
pca_vecs = ['question_vec', 'passage_vec', 'question_passage_vec', 'all_features_qa_vec']

for feature_tag, feature_vec_list in feature_vec_options.items():
    stacked_features = torch.stack(feature_vec_list)
    writer.add_embedding(stacked_features,
                         metadata=labels, tag=feature_tag, metadata_header=InstanceLabels._fields)

    if feature_tag in pca_vecs:
        stacked_features_pca = pca(stacked_features)
        writer.add_embedding(stacked_features_pca,
                             metadata=labels, tag=feature_tag + '_pca', metadata_header=InstanceLabels._fields)

writer.close()


