import numpy as np
from aluqa_or.generators.curicculum.statistics_extractor import extract_pos
from aluqa_or.generators.curicculum.helper_util import randomize_scope, randomize_total_question


class CuricculumClassC:

    def __init__(self, limit_classes=10):
        self.limit_classes = limit_classes

    def how_many_nouns(self, passage):
        context_scope, sen_index, text = randomize_scope(passage, passage_prob=0.25)

        spans, _ = extract_pos(text, pos_to_extract=['NN'])
        answer = len(spans)

        question = randomize_total_question(string_category=f'noun',
                                            context_scope=context_scope)

        sample_details = dict(
            question=question,
            answer=answer,
            spans=spans,
            span_type='SPAN'
        )

        return sample_details

    def how_many_proper_nouns(self, passage):
        context_scope, sen_index, text = randomize_scope(passage, passage_prob=0.25)

        spans, _ = extract_pos(text, pos_to_extract=['NNP'])
        answer = len(spans)

        question = randomize_total_question(string_category=f'proper noun',
                                            context_scope=context_scope)

        sample_details = dict(
            question=question,
            answer=answer,
            spans=spans,
            span_type='SPAN'
        )

        return sample_details

    def how_many_cardinals(self, passage):
        context_scope, sen_index, text = randomize_scope(passage, passage_prob=0.25)

        spans, _ = extract_pos(text, pos_to_extract=['CD'])
        answer = len(spans)

        question = randomize_total_question(string_category=f'cardinal',
                                            context_scope=context_scope)

        sample_details = dict(
            question=question,
            answer=answer,
            spans=spans,
            span_type='SPAN'
        )

        return sample_details

    def how_many_adjectives(self, passage):
        context_scope, sen_index, text = randomize_scope(passage, passage_prob=0.25)

        spans, _ = extract_pos(text, pos_to_extract=['JJ'])
        answer = len(spans)

        question = randomize_total_question(string_category=f'adjective',
                                            context_scope=context_scope)

        sample_details = dict(
            question=question,
            answer=answer,
            spans=spans,
            span_type='SPAN'
        )

        return sample_details

    def how_many_verb(self, passage):
        context_scope, sen_index, text = randomize_scope(passage, passage_prob=0.25)

        spans, _ = extract_pos(text, pos_to_extract=['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        answer = len(spans)

        question = randomize_total_question(string_category=f'verb',
                                            context_scope=context_scope)

        sample_details = dict(
            question=question,
            answer=answer,
            spans=spans,
            span_type='SPAN'
        )

        return sample_details

    def sample(self, passage):
        while True:
            class_funcs = [
                self.how_many_nouns,
                self.how_many_proper_nouns,
                self.how_many_cardinals,
                self.how_many_adjectives,
                self.how_many_verb
            ]
            sample_func = class_funcs[np.random.choice(len(class_funcs))]
            sample_details = sample_func(passage)
            if sample_details['answer'] > self.limit_classes:
                continue
            return sample_details
