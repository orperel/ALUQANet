import numpy as np
from collections import Counter
from aluqa_or.generators.curicculum.statistics_extractor import extract_pos
from aluqa_or.generators.curicculum.helper_util import randomize_scope, randomize_common_question


class CuricculumClassD:

    def __init__(self, limit_classes=10):
        self.limit_classes = limit_classes

    def _select_template(self, passage, pos_to_extract, string_category):

        order_of_common = np.random.randint(0, 3)
        order_name = ['the most common', 'the second most common', 'the third most common'][order_of_common]

        context_scope, sen_index, text = randomize_scope(passage, passage_prob=1.00)

        spans, tokens_text = extract_pos(text, pos_to_extract=pos_to_extract)
        sorted_tokens_by_frequency = Counter(tokens_text).most_common()
        most_common_token = sorted_tokens_by_frequency[order_of_common]

        answer = most_common_token[1]

        question = randomize_common_question(string_category=f'{order_name} {string_category}',
                                             context_scope=context_scope)

        sample_details = dict(
            question=question,
            answer=answer,
            spans=spans,
            span_type='SPAN'
        )

        return sample_details

    def how_many_times_most_common_noun(self, passage):
        return self._select_template(passage, ['NN'], 'noun')

    def how_many_times_most_common_proper_noun(self, passage):
        return self._select_template(passage, ['NNP'], 'proper noun')

    def how_many_times_most_common_cardinal(self, passage):
        return self._select_template(passage, ['CD'], 'cardinal')

    def how_many_times_most_common_adjective(self, passage):
        return self._select_template(passage, ['JJ'], 'adjective')

    def how_many_times_most_common_verb(self, passage):
        return self._select_template(passage, ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], 'verb')

    def sample(self, passage):
        allowed_misses = 0
        while True:
            class_funcs = [
                self.how_many_times_most_common_noun,
                self.how_many_times_most_common_proper_noun,
                self.how_many_times_most_common_cardinal,
                self.how_many_times_most_common_adjective,
                self.how_many_times_most_common_verb
            ]
            sample_func = class_funcs[np.random.choice(len(class_funcs))]
            sample_details = sample_func(passage)
            if sample_details['answer'] > self.limit_classes:
                continue
            if sample_details['answer'] == 1:   # Try to avoid cases with answer of 1 as much as possible..
                allowed_misses += 1
                if allowed_misses > 4:
                    return sample_details
                continue
            return sample_details
