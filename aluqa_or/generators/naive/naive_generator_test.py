from aluqa_or.generators.naive.naive_generator import NaiveQAGenerator


qa_generator = NaiveQAGenerator()

for idx in range(1, 11):
    question, answer, passage, \
    passage_span_indices, passage_spans, question_metadata, passage_metadata = qa_generator.generate()

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
