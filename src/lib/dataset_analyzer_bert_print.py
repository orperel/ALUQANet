from datetime import datetime
from collections import namedtuple
import torch
import string
import numpy as np
from sklearn.decomposition import PCA

from textwrap import wrap
import re
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from drop_bert.augmented_bert_plus import NumericallyAugmentedBERTPlus
from aluqa_or.aluqa_experimental_model import ALUQAExperimental
from drop_bert.data_processing import BertDropTokenizer, BertDropTokenIndexer, BertDropReader
from src.lib.inference_utils import data_instance_to_model_input, filter_count_questions
from torch.utils.tensorboard import SummaryWriter
from aluqa_itay.aluqa_count_spans_syntactic_parser import AluQACount
from aluqa_itay.data_processing import PickleReader

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


def load_aluqa_model(weights_path, selection_output_file_path=None):
    device_num = 0
    device = torch.device('cuda:%d' % device_num)
    model = AluQACount(Vocabulary(), 'bert-base-uncased', special_numbers=[100, 1], answering_abilities=["counting"])
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.selection_output_file_path = selection_output_file_path
    model.to(device)
    model.eval()

    return model


def create_aluqa_reader(data_path, question_type, max_span_length, remove_containing_spans):
    reader = PickleReader(question_type=question_type,
                          max_span_length=max_span_length,
                          remove_containing_spans=remove_containing_spans)
    instances = reader.read(data_path)

    return instances


def load_nabert_model(weights_path, use_custom_model=False):
    device_num = 0
    device = torch.device('cuda:%d' % device_num)
    if not use_custom_model:
        model = NumericallyAugmentedBERTPlus(Vocabulary(), 'bert-base-uncased', special_numbers=[100, 1])
    else:
        model = ALUQAExperimental(Vocabulary(), 'expected_count_per_sentence', 'bert-base-uncased', special_numbers=[100, 1])
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


def plot_confusion_matrix(writer, correct_labels, predict_labels, labels,
                          title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
    '''
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = plt.Figure(figsize=(7, 7), dpi=160, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    writer.add_figure(figure=fig, tag=tensor_name)


# model = load_nabert_model(weights_path='../results/nabert/best.th')
# model = load_nabert_model(weights_path='../results/aluqa_exp7/best.th', use_custom_model=True)
# instances = create_nabert_reader(data_path='/home/itaysofer/Desktop/Drop/data/drop_dataset_train.json')
model = load_aluqa_model('/home/itaysofer/Desktop/Drop/run/aluqa_span_count_overfit/serialization/best.th',
                         selection_output_file_path='./selection_output' + datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt")
instances = create_aluqa_reader(data_path='/home/itaysofer/Desktop/Drop/data/drop_dataset_spans_dev.pickle',
                                question_type=["count"],
                                max_span_length=10,
                                remove_containing_spans=True)

feature_vecs_all = []
feature_vecs_token_count = []
feature_question_vecs = []
feature_passage_vecs = []
feature_question_passage_vecs = []
feature_vecs_all_and_qa_vecs = []
bert_question_features = [[] for _ in range(12)]
bert_passage_features = [[] for _ in range(12)]
labels = []
InstanceLabels = namedtuple('InstanceLabels',
                            ['em_correct', 'f1_score', 'f1_correct', 'count_prediction',
                             'answer_types', 'answer_type_correct',
                             'question_tokens', 'answer_texts', 'passage_numbers', 'passage_tokens', 'answer_content'])

count_pred_samples = []
count_gt_samples = []


with torch.no_grad():
    for instance_idx, instance in enumerate(instances):

        entry = extract_instance_properties(instance)

        if not filter_count_questions(question_text=' '.join(entry['question_tokens']), answer_text=entry['answer_texts']):
            continue

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

        count_prediction_label = [ans['value'] for ans in prediction['answer'] if ans['answer_type'] == 'count']
        count_prediction_label = count_prediction_label[0] if len(count_prediction_label) > 0 else None

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

        count_prediction_label_idx = "Other" if count_prediction_label is None else str(count_prediction_label)
        count_pred_samples.append(count_prediction_label_idx)
        count_gt_idx = "Other" if answer_texts not in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10")  \
                       else str(answer_texts)
        count_gt_samples.append(count_gt_idx)


# Writer will output to ./tb_data_analysis/ directory
# (1) To run the TB server use:
# tensorboard --logdir=tb_data_analysis
# (2) Forward the display to local machine with:
# ssh -i <PATH_TO_REMOTE_MACHINE_SSH_KEY> -N -L localhost:6006:localhost:6006 ubuntu@<REMOTE_IP>
writer = SummaryWriter("tb_data_analysis/aluqa_count_spans" + datetime.now().strftime("%Y%m%d-%H%M%S"))

plot_confusion_matrix(writer=writer,
                      correct_labels=np.array(count_gt_samples),
                      predict_labels=np.array(count_pred_samples),
                      labels=['align_start', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Other', 'align_end'],
                      title='Count Head - Confusion matrix',
                      tensor_name='Count/ConfusionMatrix',
                      normalize=False)

writer.close()


