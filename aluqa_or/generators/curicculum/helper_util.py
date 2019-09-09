import numpy as np
from aluqa_or.generators.curicculum.statistics_extractor import extract_sentences


def randomize_scope(passage, passage_prob=0.5, eliminate_extra_passage_scopes=False):
    is_passage = np.random.binomial(n=1, p=passage_prob)
    if is_passage:
        if eliminate_extra_passage_scopes:
            context_scope = np.random.choice(['the passage',
                                              'the entire passage',
                                              'the context text'])
        else:
            context_scope = np.random.choice(['the passage',
                                              'the entire passage',
                                              'all sentences',
                                              'all sentences combined',
                                              'the context text'])
        index = -1
        text = passage
    else:
        sentences = extract_sentences(passage)
        sentence_number = np.random.random_integers(len(sentences)) % 15
        sentence_index = sentence_number - 1
        sentence = sentences[sentence_index]

        sentence_phrase_groups = [
            ['the 1st sentence', 'the 2nd sentence', 'the 3rd sentence', 'the 4th sentence', 'the 5th sentence', 'the 6th sentence', 'the 7th sentence', 'the 8th sentence', 'the 9th sentence', 'the 10th sentence', 'the 11th sentence', 'the 12th sentence', 'the 13th sentence', 'the 14th sentence', 'the 15th sentence'],
            ['the first sentence', 'the second sentence', 'the third sentence', 'the fourth sentence', 'the fifth sentence', 'the sixth sentence', 'the seventh sentence', 'the eighth sentence', 'the ninth sentence', 'the tenth sentence', 'the eleventh sentence', 'the twelfth sentence', 'the thirteenth sentence', 'the fourteenth sentence', 'the fifteenth sentence'],
            ['sentence 1', 'sentence 2', 'sentence 3', 'sentence 4', 'sentence 5', 'sentence 6', 'sentence 7', 'sentence 8', 'sentence 9', 'sentence 10', 'sentence 11', 'sentence 12', 'sentence 13', 'sentence 14', 'sentence 15'],
            ['sentence one', 'sentence two', 'sentence three', 'sentence four', 'sentence five', 'sentence six', 'sentence seven', 'sentence eight', 'sentence nine', 'sentence ten', 'sentence eleven', 'sentence twelve', 'sentence thirteen', 'sentence fourteen', 'sentence fifteen']
        ]
        sentence_phrase_group = sentence_phrase_groups[np.random.choice(len(sentence_phrase_groups))]
        sentence_phrase = sentence_phrase_group[sentence_index]

        context_scope = sentence_phrase
        index = sentence_index
        text = sentence

    return context_scope, index, text


def randomize_instances_occurance_question(string_asked_about, string_category, context_scope):

    question = np.random.choice([
        f'How many instances of {string_asked_about} are there in {context_scope}?',
        f'How many {string_asked_about} occurrences exist in {context_scope}?',
        f'How many times does the {string_category} {string_asked_about} appear in {context_scope}?',
        f'What amount of times does {string_asked_about} appear in {context_scope}?',
        f'How many times is {string_asked_about} mentioned within {context_scope} in total?',
    ])

    return question


def randomize_total_question(string_category, context_scope):

    question = np.random.choice([
        f'How many instances of {string_category}s are there in {context_scope}?',
        f'How many {string_category} occurrences exist in {context_scope}?',
        f'How many {string_category}s exist in total in {context_scope}?',
        f'How many {string_category}s are there in {context_scope}?',
        f'What amount of {string_category}s is there in total in {context_scope}?',
    ])

    return question

def randomize_common_question(string_category, context_scope):

    question = np.random.choice([
        f'What is the amount of {string_category} in {context_scope}?',
        f'How many instances in {context_scope} are there for {string_category}?',
        f'How many occurrences of {string_category} exist in {context_scope}?',
    ])

    return question

def randomize_ner_question(singular_string_category, plural_string_category, context_scope):

    question = np.random.choice([
        f'How many instances of {plural_string_category} are there in {context_scope}?',
        f'How many {singular_string_category} occurrences exist in {context_scope}?',
        f'How many {plural_string_category} appear in total in {context_scope}?',
        f'How many {plural_string_category} mentions are there in {context_scope}?',
        f'What amount of {singular_string_category} instances is there in total in {context_scope}?',
    ])

    return question
