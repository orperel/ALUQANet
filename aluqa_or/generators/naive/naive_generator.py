from aluqa_or.generators.naive.ner_tokens_generator import NERTokensGenerator
from aluqa_or.generators.naive.naive_questions_generator import NaiveQuestionsGenerator
from aluqa_or.generators.naive.naive_passages_generator import NaivePassagesGenerator

NER_FREQUENCY_TABLE_PATH = 'ner_frequencies.json'
CONTENT_SENTENCES_CORPUS_PATH = 'content_sentences.json'

ner_sampler = NERTokensGenerator(ner_frequency_table_path=NER_FREQUENCY_TABLE_PATH)
questions_generator = NaiveQuestionsGenerator(ner_sampler)
passages_generator = NaivePassagesGenerator(ner_sampler, corpus_sentences_path=CONTENT_SENTENCES_CORPUS_PATH)

for idx in range(1, 11):
    question, question_metadata = questions_generator.generate_question()
    answer, passage, passage_span_indices, passage_spans, passage_metadata = \
        passages_generator.generate_passage_and_answer(question, question_metadata)

    print('================================')
    print('Example #' + str(idx))
    print('================================')

    print('Question:')
    print('------------------------')
    print(question)
    print()

    print('Answer:')
    print('------------------------')
    print(answer)
    print()

    print('Passage:')
    print('------------------------')
    print(passage)
    print()

    print('Spans:')
    print('------------------------')
    print(passage_spans)
    print()

    print('Span indices:')
    print('------------------------')
    print(passage_span_indices)
    print()

    print('Question Metadata:')
    print('------------------------')
    print(question_metadata)
    print()

    print('Passage Metadata:')
    print('------------------------')
    print(passage_metadata)
    print()



