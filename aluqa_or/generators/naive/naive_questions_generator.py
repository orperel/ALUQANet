import numpy as np
from aluqa_or.generators.naive.ner_tokens_generator import NERTokensGenerator


class NaiveQuestionsGenerator:

    def __init__(self, ner_tokens_generator: NERTokensGenerator):
        self.ner_tokens_generator = ner_tokens_generator

    @staticmethod
    def generate_question_prefix():
        AVAILABLE_QUESTION_PREFIXES = [
            'How many',
            'What number of',
            'What amount of'
        ]

        return np.random.choice(AVAILABLE_QUESTION_PREFIXES)

    @staticmethod
    def generate_countable_object_of_interest():
        AVAILABLE_COUNTABLE_OBJECTS = [
            'goals',
            'touchdowns',
            'points',
            'field goals'
        ]

        return np.random.choice(AVAILABLE_COUNTABLE_OBJECTS)

    def generate_actor_name(self):
        return self.ner_tokens_generator.sample(category='PERSON')

    def generate_countable_verb_phrase(self):

        actor_name = self.generate_actor_name()

        AVAILABLE_PHRASES_TEMPLATES = [
            f'did {actor_name} score',
            f'did {actor_name} get',
            f'have {actor_name} scored',
            f'were scored in the game',
            f'were scored by {actor_name}',
            f'were made by {actor_name}'
        ]

        verb_phrase = np.random.choice(AVAILABLE_PHRASES_TEMPLATES)
        if actor_name not in verb_phrase:
            actor_name = None

        return verb_phrase, actor_name

    def generate_temporal_filter(self):

        FILTER_TYPE = np.random.choice(('NO_FILTER', 'EXACT', 'TIME_SPAN'), p=np.array((0.3, 0.6, 0.1)))

        if FILTER_TYPE == 'NO_FILTER':
            return '', None, None
        elif FILTER_TYPE == 'EXACT':
            min_event_time = self.ner_tokens_generator.sample(category='ORDINAL')
            max_event_time = None
            temporal_filter = 'in the ' + min_event_time + ' quarter'
            return temporal_filter, min_event_time, max_event_time
        elif FILTER_TYPE == 'TIME_SPAN':
            available_spans = [
                ('first', 'second'),
                ('first', 'third'),
                ('second', 'third'),
                ('third', 'fourth')
            ]
            min_event_time, max_event_time = available_spans[np.random.choice(len(available_spans))]
            temporal_filter = 'between the ' + min_event_time + ' and the ' + max_event_time + ' quarter'
            return temporal_filter, min_event_time, max_event_time
        else:
            raise ValueError('Invalid filter type')

    def generate_quantity_filter(self):
        FILTER_TYPE = np.random.choice(('NO_FILTER', 'QUANTITY_FILTER'), p=np.array((0.3, 0.7)))

        if FILTER_TYPE == 'NO_FILTER':
            return '', None, None, None
        else:
            quantity_phrase, numbers, comparators, units = self.ner_tokens_generator.sample_semantic_quantity()
            return quantity_phrase, numbers, comparators, units

    def generate_question(self):

        question_prefix = self.generate_question_prefix()
        question_object = self.generate_countable_object_of_interest()
        verb_phrase, actor_name = self.generate_countable_verb_phrase()

        temporal_filter, min_event_time, max_event_time = self.generate_temporal_filter()
        quantity_phrase, numbers, comparators, units = self.generate_quantity_filter()

        question = ' '.join((question_prefix, question_object, verb_phrase))
        if len(temporal_filter) > 0:
            question += ' ' + temporal_filter
        if len(quantity_phrase) > 0:
            if len(temporal_filter) > 0:
                question += ' which were'
            question += ' ' + quantity_phrase
        question += '?'

        meta_data = dict(
            actor_name=actor_name,
            question_object=question_object,
            min_event_time=min_event_time,
            max_event_time=max_event_time,
            numbers=numbers,
            comparators=comparators,
            units=units
        )

        return question, meta_data
