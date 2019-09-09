import numpy as np
from aluqa_or.generators.curicculum.statistics_extractor import extract_letters_frequency, extract_words,\
    extract_sentences, extract_passage_numbers
from aluqa_or.generators.curicculum.helper_util import randomize_instances_occurance_question, randomize_scope, \
    randomize_total_question


class CuricculumClassA:

    def __init__(self, limit_classes=10):
        self.limit_classes = limit_classes

    def how_many_times_character_appears(self, passage):
        context_scope, sen_index, text = randomize_scope(passage, passage_prob=0.5)

        letters_frequency = extract_letters_frequency(text)

        if self.limit_classes is not None:
            letters_frequency = dict(filter(lambda entry: entry[1] < self.limit_classes, letters_frequency.items()))

        random_character = np.random.choice(list(letters_frequency.keys()))
        answer = letters_frequency[random_character]
        start_idx = passage.index(text)
        spans = [(start_idx + i, start_idx + i + 1) for i, x in enumerate(text) if x == random_character]

        question = randomize_instances_occurance_question(string_asked_about=f'\'{random_character}\'',
                                                          string_category='character',
                                                          context_scope=context_scope)

        sample_details = dict(
            question=question,
            answer=answer,
            spans=spans,
            span_type='CHAR'
        )

        return sample_details

    def how_many_words_in_total(self, passage):
        context_scope, sen_index, text = randomize_scope(passage, passage_prob=0.0)

        words = extract_words(text)
        start_idx = passage.index(text)
        spans = [(start_idx + w.idx, start_idx + w.idx + len(str(w))) for w in words]
        answer = len(words)

        question = randomize_total_question(string_category='word',
                                            context_scope=context_scope)

        sample_details = dict(
            question=question,
            answer=answer,
            spans=spans,
            span_type='WORD'
        )

        return sample_details

    def how_many_title_case_words_in_total(self, passage):
        context_scope, sen_index, text = randomize_scope(passage, passage_prob=0.5)

        words = extract_words(text)
        title_words = list(filter(lambda tok: len(tok.text) > 0 and tok.text[0].isupper(), words))
        start_idx = passage.index(text)
        spans = [(start_idx + w.idx, start_idx + w.idx + len(str(w))) for w in title_words]
        answer = len(title_words)

        question = randomize_total_question(string_category='title case word',
                                            context_scope=context_scope)

        sample_details = dict(
            question=question,
            answer=answer,
            spans=spans,
            span_type='WORD'
        )

        return sample_details

    def how_many_sentences_in_total(self, passage):
        context_scope, sen_index, passage = randomize_scope(passage, passage_prob=1.0)

        sentences = extract_sentences(passage)
        spans = [(passage.index(s), passage.index(s) + len(s)) for s in sentences]
        answer = len(sentences)

        question = randomize_total_question(string_category='sentence',
                                            context_scope=context_scope)

        sample_details = dict(
            question=question,
            answer=answer,
            spans=spans,
            span_type='SENTENCE'
        )

        return sample_details

    def how_many_numbers(self, passage):
        context_scope, sen_index, text = randomize_scope(passage, passage_prob=0.5)

        start_idx = passage.index(text)
        num_words, num_indices = extract_passage_numbers(text)
        spans = [(start_idx + idx, start_idx + idx + len(str(w))) for w, idx in zip(num_words, num_indices)]
        answer = len(num_words)

        question = randomize_total_question(string_category='number',
                                            context_scope=context_scope)

        sample_details = dict(
            question=question,
            answer=answer,
            spans=spans,
            span_type='WORD'
        )

        return sample_details

    def sample(self, passage):
        while True:
            class_funcs = [
                self.how_many_times_character_appears,
                self.how_many_words_in_total,
                self.how_many_sentences_in_total,
                self.how_many_title_case_words_in_total,
                self.how_many_numbers
            ]
            sample_func = class_funcs[np.random.choice(len(class_funcs))]
            sample_details = sample_func(passage)
            if sample_details['answer'] > self.limit_classes:
                continue
            return sample_details
