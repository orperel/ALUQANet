import json
import numpy as np


class NERTokensGenerator:

    def __init__(self, ner_frequency_table_path):
        self.ner_frequency_table = self._load_ner_frequency_table(ner_frequency_table_path)

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
        numbers = [int(s) for s in ner_sample.split() if s.isdigit()]

        if len(numbers) == 1:
            if any([term in ner_sample for term in ('less', 'lower', 'small', 'short', 'below')]):
                comparators = ['<']
            elif any([term in ner_sample for term in ('longer', 'more', 'higher', 'big', 'great', 'above')]):
                comparators = ['>']
            else:
                comparators = ['=']
        else:
            comparators = ['>', '<']  # Assume between

        units = None
        if 'yard' in ner_sample:
            units = 'yard'
        elif 'yards' in ner_sample:
            units = 'yards'
        elif 'points' in ner_sample:
            units = 'points'
        elif 'meters' in ner_sample:
            units = 'meters'

        return ner_sample, numbers, comparators, units

    def sample(self, category):
        phrases, probabilities = self.ner_frequency_table[category]
        ner_sample = np.random.choice(phrases, p=probabilities)
        return ner_sample
