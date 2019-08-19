from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import nltk


class SpanExtractor:
    def __init__(self):
        archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
        self.predictor = Predictor.from_archive(archive, 'constituency-parser')

    def extract_passage_spans(self, passage_tokens, spans_labels, max_span_length):
        spans = []
        word_tokens_text = [token.text for token in passage_tokens]
        sentences_indices = self._get_sentence_indices(word_tokens_text)
        for i in range(len(sentences_indices) - 1):
            sentence_tokens = word_tokens_text[sentences_indices[i]: sentences_indices[i + 1]]
            sentence_spans = self.extract_sentence_spans(sentence_tokens, spans_labels, max_span_length)
            sentence_spans_shifted = [tuple([span[0] + sentences_indices[i], span[1] + sentences_indices[i]])
                                      for span in sentence_spans]
            spans += sentence_spans_shifted
        return spans

    def extract_sentence_spans(self, sentence_tokens, spans_labels, max_span_length):
        fixed_sentence_tokens = self.fix_sentence_tokens(sentence_tokens)
        prediction = self.predictor.predict_json({"sentence": ' '.join(fixed_sentence_tokens)})

        syntactic_tree = nltk.Tree.fromstring(prediction['trees'])
        syntactic_tree = self._add_indices_to_terminals(syntactic_tree)

        spans_subtrees = list(syntactic_tree.subtrees(filter=lambda x: x.label() in spans_labels))

        spans_ngrams = list(map(lambda t: t.leaves(), spans_subtrees))
        spans_ngrams = list(filter(lambda ngram: len(ngram) < max_span_length, spans_ngrams))

        spans = []
        for span_ngram in spans_ngrams:
            start_index = int(span_ngram[0].split('_')[-1])
            end_index = int(span_ngram[-1].split('_')[-1]) + 1
            spans += [(start_index, end_index)]

        spans = self.remove_contained_spans(spans)

        return spans

    @staticmethod
    def fix_sentence_tokens(sentence_tokens):
        # replace round parenthesis with square ones, as round ones are used for tree syntax
        fixed_sentence_tokens = list(
            map(lambda token: "[" if token is "(" else "]" if token is ")" else token, sentence_tokens))

        # remove remaining round parenthesis from tokens too
        fixed_sentence_tokens = [token.replace('(', '').replace(')', '') for token in fixed_sentence_tokens]
        return fixed_sentence_tokens

    @staticmethod
    def _get_sentence_indices(word_tokens_text):
        sentences_indices = [0] + [index + 1 for index, value in enumerate(word_tokens_text) if
                                   value == "." or value == "!" or value == "?"]
        if sentences_indices[-1] != len(word_tokens_text):
            sentences_indices += [len(word_tokens_text)]
        return sentences_indices

    @staticmethod
    def _add_indices_to_terminals(tree):
        for idx, _ in enumerate(tree.leaves()):
            tree_location = tree.leaf_treeposition(idx)
            non_terminal = tree[tree_location[:-1]]
            non_terminal[0] = non_terminal[0] + "_" + str(idx)
        return tree

    @staticmethod
    def remove_contained_spans(spans):
        if len(spans) == 0:
            return spans

        filtered_spans = spans[:1]
        for span in spans[1:]:
            if SpanExtractor.contained_in(span, filtered_spans[-1]) is False:
                filtered_spans += [span]

        return filtered_spans

    @staticmethod
    def contained_in(inner_span, outer_span):
        return inner_span[0] >= outer_span[0] and inner_span[1] <= outer_span[1]
