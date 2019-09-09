import numpy as np
from aluqa_or.generators.curicculum.statistics_extractor import extract_ner
from aluqa_or.generators.curicculum.helper_util import randomize_scope, randomize_ner_question


class CuricculumClassE:

    def __init__(self, limit_classes=10):
        self.limit_classes = limit_classes

    def _how_many_template(self, passage, ner_categories, singular_string_category, plural_string_category):
        context_scope, sen_index, text = randomize_scope(passage, passage_prob=0.85)

        ner_and_spans = extract_ner(text, ner_categories)
        spans = [(entry[1], entry[2]) for entry in ner_and_spans]
        answer = len(spans)

        question = randomize_ner_question(singular_string_category=singular_string_category,
                                          plural_string_category=plural_string_category,
                                          context_scope=context_scope)

        sample_details = dict(
            question=question,
            answer=answer,
            spans=spans,
            span_type='SPAN'
        )

        return sample_details

    def how_many_persons(self, passage):
        return self._how_many_template(passage, ['PERSON'], 'person', 'people')

    def how_many_organizations(self, passage):
        return self._how_many_template(passage, ['ORG'], 'organization', 'organizations')

    def how_many_time_intervals(self, passage):
        return self._how_many_template(passage, ['TIME', 'DATE'], 'time span', 'temporal entities')

    def how_many_locations(self, passage):
        return self._how_many_template(passage, ['GPE', 'LOC', 'FAC'], 'location', 'locations')

    def sample(self, passage):
        while True:
            class_funcs = [
                self.how_many_persons,
                self.how_many_organizations,
                self.how_many_time_intervals,
                self.how_many_locations
            ]
            sample_func = class_funcs[np.random.choice(len(class_funcs))]
            sample_details = sample_func(passage)
            if sample_details['answer'] > self.limit_classes:
                continue
            return sample_details
