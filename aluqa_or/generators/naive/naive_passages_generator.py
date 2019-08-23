import numpy as np
import json
import random
from collections import defaultdict
from aluqa_or.generators.naive.ner_tokens_generator import NERTokensGenerator
from aluqa_or.generators.naive.pcfg_generator import PCFGGenerator


class NaivePassagesGenerator:

    def __init__(self, ner_tokens_generator: NERTokensGenerator,
                 corpus_sentences_path, pcfg_grammer_path, use_only_nfl_passages_as_noise=True):
        self.ner_tokens_generator = ner_tokens_generator

        self.ner_tokens_generator = ner_tokens_generator
        self.corpus = self._load_corpus(corpus_sentences_path)
        self.possible_placeholders = self.ner_tokens_generator.available_ner_categories() + ['GOAL']
        self.pcfg = PCFGGenerator(pcfg_grammer_path)

        self.use_only_nfl_passages_as_noise = use_only_nfl_passages_as_noise

    @staticmethod
    def _load_corpus(content_sentences_path):
        with open(content_sentences_path) as json_file:
            content_sentences = json.load(json_file)

        # Filter out empty passages
        content_sentences = {k: v for k, v in content_sentences.items() if len(v) > 0}
        return content_sentences

    def _seed_passage(self):
        seeds = list(self.corpus.keys())
        if self.use_only_nfl_passages_as_noise:
            seeds = list(filter(lambda passage_name: passage_name.startswith('nfl'), seeds))
        passage_name = np.random.choice(seeds)

        return self.corpus[passage_name]

    def _seed_ner_pool(self):
        ner_pool = defaultdict(list)
        for ner_category in self.ner_tokens_generator.available_ner_categories():
            for _ in range(2):
                ner_value = self.ner_tokens_generator.sample(category=ner_category)
                ner_pool[ner_category].append(ner_value)

        return ner_pool

    def _fill_sentence_with_named_entities(self, sentence, metadata):
        for tag in self.possible_placeholders:
            next_idx = str.find(sentence, tag)
            while next_idx != -1:
                prefix = sentence[:next_idx]
                suffix = sentence[next_idx + len(tag):]
                implemented_value = self.ner_tokens_generator.sample(category=tag)
                sentence = prefix + implemented_value + suffix
                next_idx = str.find(sentence, tag)

        return sentence

    # def _find_sentence_with_skeleton(self, metadata):
    #
    #     requirements = []
    #     if metadata['actor_name'] is not None:
    #         requirements.append(['[PERSON]'])
    #     if metadata['question_object'] is not None:
    #         requirements.append(['goal', 'touchdown', 'score'])
    #     if metadata['min_event_time'] is not None:
    #         requirements.append(['quarter'])
    #     if metadata['numbers'] is not None:
    #         requirements.append(['[ORDINAL]', '[CARDINAL]'])
    #
    #     all_sentences = [item[0] for sublist in [list(sentence.values())
    #                                           for sentence in list(self.corpus.values())] for item in sublist]
    #
    #     candidates = list()
    #     for sentence in all_sentences:
    #         for requirement in requirements:
    #             fulfilled = False
    #             for option in requirement:
    #                 if option in sentence:
    #                     fulfilled = True
    #                     break
    #             if not fulfilled:
    #                 break
    #         if fulfilled:
    #             candidates.append(sentence)
    #
    #     return candidates

    @staticmethod
    def _sample_quantitive_range(metadata):
        goal_amount_entry = metadata['[GoalAmount]']
        low = goal_amount_entry[0] if goal_amount_entry is not None and len(goal_amount_entry) >= 1 else 1
        high = goal_amount_entry[1] if goal_amount_entry is not None and len(goal_amount_entry) >= 2 else low + 10
        return str(np.random.randint(low=low, high=high + 1))

    @staticmethod
    def _sample_temporal_range(metadata):
        low = metadata['[MinTemporal]'] if metadata['[MinTemporal]'] is not None else 'first'
        high = metadata['[MaxTemporal]'] if metadata['[MaxTemporal]'] is not None else low

        # low must be higher every time we sample it
        should_increment_low = 0
        if '[Temporal]' in metadata['implemented_values']:
            implemented_tuples = metadata['implemented_values']['[Temporal]']
            implemented_tuple = implemented_tuples[-1]  # Take the last value implemented
            low = implemented_tuple[0]
            should_increment_low = 1

        temporal_options = ['first', 'second', 'third', 'fourth']
        max_idx = len(temporal_options) - 1
        low_idx = min(temporal_options.index(low) + should_increment_low, max_idx)
        high_idx = temporal_options.index(high)
        # high_idx must be higher than low_idx but no more than temporal_options's last index
        high_idx = max(high_idx + 1, min(low_idx + 2, max_idx))
        temporal_options = temporal_options[low_idx:high_idx]
        if len(temporal_options) == 0:
            temporal_options = ['fourth']   # Can't be empty
        temporal_val = np.random.choice(temporal_options)

        return temporal_val

    def _sample_actor_name(self, metadata):

        actor_name = metadata['[Actor]'] if metadata['[Actor]'] is not None \
            else self.ner_tokens_generator.sample(category='[PERSON]')

        actor_name_components = actor_name.split()
        actor_name_variations = actor_name_components + [actor_name]
        return np.random.choice(actor_name_variations)

    def _synthesize_golden_sentence(self, metadata):
        amount_of_temporal_tags = len([t for t in ('[MaxTemporal]', '[MinTemporal]') if metadata[t] is not None])
        amount_of_temporal_tags = None if amount_of_temporal_tags is 0 else amount_of_temporal_tags
        conditions = {
            '[Temporal]': amount_of_temporal_tags
        }
        template_sentence = self.pcfg.generate(conditions)
        placeholders = [tag for tag in template_sentence.split(' ')
                        if tag.startswith('[') and ']' in tag]
        placeholders = [tag_part for tag in placeholders for tag_part in tag.split('-') if tag_part]
        placeholders = [tag_part for tag in placeholders for tag_part in tag.split('.') if tag_part]
        placeholders = [tag_part for tag in placeholders for tag_part in tag.split(',') if tag_part]
        placeholders = [tag[:-1] if tag.endswith('s') else tag for tag in placeholders]
        metadata['implemented_values'] = defaultdict(list)
        golden_sentence = template_sentence
        for tag in placeholders:
            next_idx = str.find(golden_sentence, tag)
            prefix = golden_sentence[:next_idx]
            suffix = golden_sentence[next_idx + len(tag):]

            if tag == '[Actor]':
                implemented_value = self._sample_actor_name(metadata)
            elif tag == '[GoalAmount]':
                implemented_value = self._sample_quantitive_range(metadata)
            elif tag == '[Temporal]':
                implemented_value = self._sample_temporal_range(metadata)
            else:
                implemented_value = metadata[tag]
            implemented_start = next_idx
            implemented_end = next_idx + len(implemented_value)
            metadata['implemented_values'][tag].append((implemented_value, implemented_start, implemented_end))
            golden_sentence = prefix + implemented_value + suffix

        span_indices = []
        spans = []
        goal_amnts = metadata['implemented_values']['[GoalAmount]']
        goal_units = metadata['implemented_values']['[GoalUnit]']
        question_objects = metadata['implemented_values']['[QuestionObject]']
        curr_question_object_idx = 0
        for span_idx, goal_amnt in enumerate(goal_amnts):

            span_start = goal_amnt[1]
            goal_unit = goal_units[span_idx]
            quest_object = question_objects[curr_question_object_idx]
            if quest_object[1] == goal_unit[2] + 1:
                span_end = quest_object[2]
                curr_question_object_idx += 1
            else:
                span_end = goal_unit[2]
            span_indices.append((span_start, span_end))
            spans.append(golden_sentence[span_start:span_end])

        return golden_sentence, spans, span_indices, template_sentence

    def _generate_answer(self):
        return np.random.choice([1, 2, 3, 4, 5, 6, 7], p=[0.35, 0.25, 0.12, 0.1, 0.08, 0.08, 0.02])

    def _generate_passage(self, question_metadata, answer):

        total_countable_entities = 0
        golden_set = list()
        passage_metadata = dict(template_golden_sentences=[], golden_sentences=[],
                                template_noisy_sentences=[], noisy_entities_pool=[],
                                noisy_sentences=[])
        while total_countable_entities < answer:
            golden_sentence, spans, span_indices, template_sentence = self._synthesize_golden_sentence(question_metadata)
            if total_countable_entities + len(spans) > answer:
                continue    # Too many spans, regenerate..
            else:
                golden_set.append((golden_sentence, spans, span_indices))
                total_countable_entities += len(spans)
                passage_metadata['template_golden_sentences'].append(template_sentence)
                passage_metadata['golden_sentences'].append(golden_sentence)

        ner_pool = self._seed_ner_pool()
        noisy_sentences = []
        while len(noisy_sentences) < 2:
            sentences_seed = self._seed_passage()
            noisy_sentences = []
            for sentence_order, sentence in sentences_seed.items():
                sentence = sentence[0]
                filled_sentence = self._fill_sentence_with_named_entities(sentence, ner_pool)
                noisy_sentences.append(filled_sentence)
                passage_metadata['template_noisy_sentences'].append(sentence)
                passage_metadata['noisy_sentences'].append(filled_sentence)
        passage_metadata['noisy_entities_pool'] = ner_pool

        passage_spans = []
        passage_span_indices = []
        passage_sentences = golden_set + noisy_sentences[1:-1]
        random.shuffle(passage_sentences)
        passage_sentences = [noisy_sentences[0]] + passage_sentences + [noisy_sentences[-1]]

        total_accumulated_len = 0
        passage = ''
        for single_sentence in passage_sentences:
            if isinstance(single_sentence, tuple):
                text = single_sentence[0]
                passage += ' ' + text
                for span in single_sentence[2]:
                    span_start = total_accumulated_len + span[0]
                    span_end = total_accumulated_len + span[1]
                    passage_span_indices.append((span_start, span_end))
                    passage_spans.append(passage[span_start:span_end])
            else:
                text = single_sentence
                passage += ' ' + text
            total_accumulated_len += len(text)

        return passage, passage_span_indices, passage_spans, passage_metadata

    def generate_passage_and_answer(self, question_metadata):
        answer = self._generate_answer()
        passage, passage_span_indices, passage_spans, passage_metadata = self._generate_passage(question_metadata, answer)

        return answer, passage, passage_span_indices, passage_spans, passage_metadata
