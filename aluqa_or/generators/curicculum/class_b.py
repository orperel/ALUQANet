import numpy as np
from aluqa_or.generators.curicculum.statistics_extractor import extract_letters_frequency, extract_words
from aluqa_or.generators.curicculum.helper_util import randomize_scope, \
    randomize_total_question


class CuricculumClassB:

    def __init__(self, limit_classes=10):
        self.limit_classes = limit_classes

    def how_many_times_vowels_appears(self, passage):
        context_scope, sen_index, text = randomize_scope(passage, passage_prob=0.0)

        letters_frequency = extract_letters_frequency(text)

        if self.limit_classes is not None:
            letters_frequency = dict(filter(lambda entry: entry[1] < self.limit_classes, letters_frequency.items()))

        vowel_entries = dict(filter(lambda e: e[0].lower() in ('a', 'e', 'i', 'u', 'o'), letters_frequency.items()))
        answer = sum(vowel_entries.values())
        start_idx = passage.index(text)
        spans = [(start_idx + i, start_idx + i + 1) for i, x in enumerate(text) if x in vowel_entries.keys()]

        question = randomize_total_question(string_category='vowel character', context_scope=context_scope)

        sample_details = dict(
            question=question,
            answer=answer,
            spans=spans,
            span_type='CHAR'
        )

        return sample_details

    def how_many_words_of_length_in_total(self, passage):
        context_scope, sen_index, text = randomize_scope(passage, passage_prob=0.0)

        words = extract_words(text)
        target_len = np.random.random_integers(9)
        target_words = list(filter(lambda tok: len(str(tok[0])) == target_len, words))
        start_idx = passage.index(text)
        spans = [(start_idx + w.idx, start_idx + w.idx + len(str(w))) for w in target_words]
        answer = len(target_words)

        question = randomize_total_question(string_category=f'{target_len} characters long word',
                                            context_scope=context_scope)

        sample_details = dict(
            question=question,
            answer=answer,
            spans=spans,
            span_type='WORD'
        )

        return sample_details

    def how_many_words_longer_than(self, passage):
        context_scope, sen_index, text = randomize_scope(passage, passage_prob=0.2)

        words = extract_words(text)
        target_len = np.random.random_integers(4, 9)
        target_words = list(filter(lambda tok: len(str(tok[0])) > target_len, words))
        start_idx = passage.index(text)
        spans = [(start_idx + w.idx, start_idx + w.idx + len(str(w))) for w in target_words]
        answer = len(target_words)

        question = randomize_total_question(string_category=f'words longer than {target_len} character',
                                            context_scope=context_scope)

        sample_details = dict(
            question=question,
            answer=answer,
            spans=spans,
            span_type='WORD'
        )

        return sample_details

    def how_many_words_shorter_than(self, passage):
        context_scope, sen_index, text = randomize_scope(passage, passage_prob=0.2)

        words = extract_words(text)
        target_len = np.random.random_integers(5)
        target_words = list(filter(lambda tok: len(str(tok[0])) < target_len, words))
        start_idx = passage.index(text)
        spans = [(start_idx + w.idx, start_idx + w.idx + len(str(w))) for w in target_words]
        answer = len(target_words)

        question = randomize_total_question(string_category=f'words shorter than {target_len} character',
                                            context_scope=context_scope)

        sample_details = dict(
            question=question,
            answer=answer,
            spans=spans,
            span_type='WORD'
        )

        return sample_details

    def how_many_words_inbetween_than(self, passage):
        context_scope, sen_index, text = randomize_scope(passage, passage_prob=0.2)

        words = extract_words(text)
        target_lower_len = np.random.random_integers(8)
        target_higher_len = np.random.random_integers(target_lower_len+1, target_lower_len+3)
        target_words = list(filter(lambda tok: target_lower_len <= len(str(tok[0])) <= target_higher_len, words))
        start_idx = passage.index(text)
        spans = [(start_idx + w.idx, start_idx + w.idx + len(str(w))) for w in target_words]
        answer = len(target_words)

        question = randomize_total_question(string_category=f'words with at least {target_lower_len} '
                                                            f'but no more than {target_higher_len} character',
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
                self.how_many_times_vowels_appears,
                self.how_many_words_of_length_in_total,
                self.how_many_words_longer_than,
                self.how_many_words_shorter_than,
                self.how_many_words_longer_than,
                self.how_many_words_inbetween_than
            ]
            sample_func = class_funcs[np.random.choice(len(class_funcs))]
            sample_details = sample_func(passage)
            if sample_details['answer'] > self.limit_classes:
                continue
            return sample_details
