import json
import numpy as np


class NERTokensGenerator:

    def __init__(self, ner_frequency_table_path):
        self.ner_frequency_table = self._load_ner_frequency_table(ner_frequency_table_path)

        goal_values = [str(val) for val in list(range(2,70))]
        goal_probabilities = [float(1 / len(goal_values)) for _ in goal_values]
        self.ner_frequency_table['GOAL'] = (goal_values, goal_probabilities)

    @staticmethod
    def _load_ner_frequency_table(ner_frequency_table_path):
        with open(ner_frequency_table_path) as json_file:
            ner_frequency_table_statistics = json.load(json_file)

        ner_frequency_table = dict()
        for ner_category, ner_statistics in ner_frequency_table_statistics.items():
            phrases = [entry[0] for entry in ner_statistics]
            probabilities = [entry[1] for entry in ner_statistics]
            ner_frequency_table[ner_category] = (phrases, probabilities)

        return ner_frequency_table

    def sample_semantic_quantity(self):
        ner_sample = self.sample(category='QUANTITY')

        valid_sample = False
        while not valid_sample:
            while not any([u in ner_sample for u in ('yard', 'points', 'meters')]):
                ner_sample = self.sample(category='QUANTITY')
                ner_sample.replace(',', '')
                ner_sample.replace('.', '')
                ner_sample.replace('-', '')
                ner_sample.replace('–', '')
            numbers = [int(s) for s in ner_sample.split() if s.isdigit()]

            if any([n > 200 for n in numbers]):
                valid_sample = False
                ner_sample = self.sample(category='QUANTITY')
                continue

            if len(numbers) == 1:
                if any([term in ner_sample for term in ('less', 'lower', 'small', 'short', 'below')]):
                    comparators = ['<']
                    valid_sample = True
                elif any([term in ner_sample for term in ('longer', 'more', 'higher', 'big', 'great', 'above')]):
                    comparators = ['>']
                    valid_sample = True
                else:
                    comparators = ['=']
                    valid_sample = False
            else:
                comparators = ['>', '<']  # Assume between
                valid_sample = True

            if not valid_sample:
                ner_sample = self.sample(category='QUANTITY')

        if 'yard' in ner_sample:
            units = 'yard'
        elif 'yards' in ner_sample:
            units = 'yards'
        elif 'points' in ner_sample:
            units = 'points'
        elif 'meters' in ner_sample:
            units = 'meters'
        else:
            units = 'yard'

        return ner_sample, numbers, comparators, units

    def available_ner_categories(self):
        return ['[' + ner_tag + ']' for ner_tag in self.ner_frequency_table.keys()]

    @staticmethod
    def _strip_category_brackets(category):
        if category.startswith('[') and category.endswith(']'):
            return category[1:-1]
        else:
            return category

    @classmethod
    def sample_from(cls, table, category):
        phrases, probabilities = table[category]
        ner_sample = np.random.choice(phrases, p=probabilities)
        return ner_sample

    def sample(self, category, strip_determiners=False):
        category = self._strip_category_brackets(category)
        ner_sample = self.sample_from(table=self.ner_frequency_table, category=category)
        if strip_determiners:
            ner_sample = ner_sample[4:] if ner_sample.lower().startswith('the ') else ner_sample
        return ner_sample
