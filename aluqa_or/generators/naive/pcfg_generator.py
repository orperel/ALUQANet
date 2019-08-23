import numpy as np
from collections import defaultdict


class PCFGGenerator:

    def __init__(self, grammer_path):
        self.grammer = self._load_grammer(grammer_path)

    def _load_grammer(self, grammer_path):
        with open(grammer_path) as grammer_file:
            grammer_rules_text = grammer_file.readlines()
            grammer_rules_text = [line for line in grammer_rules_text
                                  if not line.startswith('#') and len(line.strip()) > 0]
            grammer_rules = [[t for t in line.split()] for line in grammer_rules_text]
            grammer = defaultdict(dict)
            for rule in grammer_rules:
                weight = float(rule[0])
                src = rule[1]
                target = rule[2:]

                if 'p' not in grammer[src]:
                    grammer[src]['p'] = list()
                    grammer[src]['target'] = list()

                grammer[src]['p'].append(weight)
                grammer[src]['target'].append(target)

        for rule in grammer.values():
            total_weight = sum(rule['p'])
            rule['p'] = [float(p) / total_weight for p in rule['p']]

        return grammer

    def generate_phrase(self):
        remaining_branches = ['ROOT']
        phrase = ''

        while len(remaining_branches) > 0:
            token = remaining_branches.pop()

            if not token in self.grammer:
                phrase = token + ' ' + phrase
            else:
                token_entry = self.grammer[token]
                next_tokens_index = np.random.choice(a=list(range(len(token_entry['target']))), p=token_entry['p'])
                next_tokens = token_entry['target'][next_tokens_index]
                remaining_branches.extend(next_tokens)

        phrase = phrase.strip()
        phrase = phrase[0].title() + phrase[1:-1].strip() + phrase[-1]
        return phrase

    @staticmethod
    def verify_phrase_fulfills_conditions(phrase, conditions):
        # Make sure each amount of tags is contained exactly in the generated phrase
        for conditional_tag, conditioned_amount in conditions.items():
            if conditioned_amount is None:
                continue
            if phrase.count(conditional_tag) != conditioned_amount:
                return False
        return True

    def generate(self, conditions=None):
        if conditions is None:
            conditions = dict()
        phrase = self.generate_phrase()
        while not self.verify_phrase_fulfills_conditions(phrase, conditions):
            phrase = self.generate_phrase()
        return phrase
