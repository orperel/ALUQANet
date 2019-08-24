import json
import re
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

    @staticmethod
    def _extract_numbers_from_sample(ner_sample):
        number_strings = re.findall(r'\d+(?:[\d.,-]*\d)', ner_sample)
        numbers = [float(n) if '.' in n else int(re.sub('[,-]', '', n)) for n in number_strings]
        return numbers

    def sample_semantic_quantity(self, units):

        while True:
            ner_sample = self.sample(category='QUANTITY')
            if not any([u in ner_sample for u in units]):
                continue    # No units ; Resample

            numbers = self._extract_numbers_from_sample(ner_sample)
            if len(numbers) == 1:
                if any([term in ner_sample for term in ('less', 'lower', 'small', 'short', 'below')]):
                    comparators = ['<']
                elif any([term in ner_sample for term in ('longer', 'more', 'higher', 'big', 'great', 'above')]):
                    comparators = ['>']
                else:
                    continue    # comparators = ['='] ; Resample
            elif len(numbers) == 2:
                comparators = ['>', '<']  # Assume between
                if numbers[1] <= numbers[0]:
                    continue
            else:
                continue    # Too many numbers ; Resample
            break

        # Check which unit the numbers describe, if none exists assign the first possibility by default
        unit = None
        for candidate_unit in units:
            if candidate_unit in ner_sample:
                unit = candidate_unit
                break
        if unit is None:
            unit = units[0]

        return ner_sample, numbers, comparators, unit

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
