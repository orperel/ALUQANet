import numpy as np
from collections import Counter
from aluqa_or.generators.curicculum.statistics_extractor import extract_ner
from aluqa_or.generators.curicculum.helper_util import randomize_scope, randomize_common_question


class CuricculumClassF:

    def __init__(self, limit_classes=10):
        self.limit_classes = limit_classes

    def _how_many_common_template(self, passage, ner_categories, singular_string_category, plural_string_category):
        order_of_common = np.random.randint(0, 3)
        order_name = ['the most common', 'the second most common', 'the third most common'][order_of_common]

        context_scope, sen_index, text = randomize_scope(passage, passage_prob=1.00)

        ner_and_spans = extract_ner(text, ner_categories)
        spans = [(entry[1], entry[2]) for entry in ner_and_spans]
        tokens_text = [entry[0] for entry in ner_and_spans]
        sorted_tokens_by_frequency = Counter(tokens_text).most_common()
        most_common_token = sorted_tokens_by_frequency[order_of_common]
        answer = most_common_token[1]

        question = randomize_common_question(string_category=f'{order_name} {singular_string_category}',
                                             context_scope=context_scope)

        sample_details = dict(
            question=question,
            answer=answer,
            spans=spans,
            span_type='SPAN'
        )

        return sample_details

    def how_many_most_common_persons(self, passage):
        return self._how_many_common_template(passage, ['PERSON'], 'person', 'people')

    def how_many_most_common_organizations(self, passage):
        return self._how_many_common_template(passage, ['ORG'], 'organization', 'organizations')

    def how_many_most_common_locations(self, passage):
        return self._how_many_common_template(passage, ['GPE', 'LOC', 'FAC'], 'location', 'locations')

    def sample(self, passage):
        allowed_misses = 0
        while True:
            class_funcs = [
                self.how_many_most_common_persons,
                self.how_many_most_common_organizations,
                self.how_many_most_common_locations
            ]
            sample_func = class_funcs[np.random.choice(len(class_funcs))]
            sample_details = sample_func(passage)
            if sample_details['answer'] > self.limit_classes:
                continue
            if sample_details['answer'] == 1:   # Try to avoid cases with answer of 1 as much as possible..
                allowed_misses += 1
                if allowed_misses > 3:
                    return sample_details
                continue
            return sample_details
