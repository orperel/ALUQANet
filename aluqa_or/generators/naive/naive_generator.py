from aluqa_or.generators.naive.ner_tokens_generator import NERTokensGenerator
from aluqa_or.generators.naive.naive_questions_generator import NaiveQuestionsGenerator


ner_sampler = NERTokensGenerator(ner_frequency_table_path='ner_frequencies_small.json')
questions_generator = NaiveQuestionsGenerator(ner_sampler)

for _ in range(10):
    question, metadata = questions_generator.generate_question()
    print(question)
    print(metadata)
