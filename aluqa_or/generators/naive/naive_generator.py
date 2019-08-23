from aluqa_or.generators.naive.ner_tokens_generator import NERTokensGenerator
from aluqa_or.generators.naive.naive_questions_generator import NaiveQuestionsGenerator
from aluqa_or.generators.naive.naive_passages_generator import NaivePassagesGenerator


NER_FREQUENCY_TABLE_PATH = 'resources/ner_frequencies.json'
CONTENT_SENTENCES_CORPUS_PATH = 'resources/content_sentences.json'
PCFC_GRAMMER_PATH = 'resources/naive_golden_sentences_grammer.txt'


class NaiveQAGenerator:

    def __init__(self, use_only_nfl_passages_as_noise=True):
        ner_sampler = NERTokensGenerator(ner_frequency_table_path=NER_FREQUENCY_TABLE_PATH)
        self.questions_generator = NaiveQuestionsGenerator(ner_sampler)
        self.passages_generator = NaivePassagesGenerator(ner_sampler,
                                                         corpus_sentences_path=CONTENT_SENTENCES_CORPUS_PATH,
                                                         pcfg_grammer_path=PCFC_GRAMMER_PATH,
                                                         use_only_nfl_passages_as_noise=use_only_nfl_passages_as_noise)

    def generate(self):
        question, question_metadata = self.questions_generator.generate_question()
        answer, passage, passage_span_indices, passage_spans, passage_metadata = \
            self.passages_generator.generate_passage_and_answer(question_metadata)

        return question, answer, passage, passage_span_indices, passage_spans, question_metadata, passage_metadata

