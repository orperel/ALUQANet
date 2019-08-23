import numpy as np
from aluqa_or.generators.naive.ner_tokens_generator import NERTokensGenerator
from aluqa_or.generators.naive.random_sentence_generator import RandomSentenceGenerator


class NaivePassagesGenerator:

    def __init__(self, ner_tokens_generator: NERTokensGenerator, corpus_sentences_path):
        self.ner_tokens_generator = ner_tokens_generator
        self.random_sentences_generator = RandomSentenceGenerator(ner_tokens_generator, corpus_sentences_path)

    def generate_answer(self):
        return np.random.choice([1, 2, 3, 4, 5, 6, 7], p=[0.35, 0.25, 0.12, 0.1, 0.08, 0.08, 0.02])

    def sample_amount_of_sentences(self):
        np.random.choice([7, 8, 9, 10, 11, 12, 13])

    def generate_passage_and_answer(self, question, metadata):
        answer = self.generate_answer()
        passage, passage_span_indices, passage_spans, passage_metadata =\
            self.random_sentences_generator.sample(metadata, answer)

        return answer, passage, passage_span_indices, passage_spans, passage_metadata
