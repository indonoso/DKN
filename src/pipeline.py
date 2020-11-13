from transformers import FeatureExtractionPipeline
import numpy as np


class CachedFeatureExtractionPipeline(FeatureExtractionPipeline):

    def __init__(self, word_dim, max_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings_cache = {}
        self.word_dim = word_dim
        self.max_length = max_length

    def _parse_and_tokenize(self, inputs, **kwargs):
        """
        Parse arguments and tokenize
        """
        # Parse arguments
        inputs = self.tokenizer(
            inputs,
            return_tensors=self.framework,
            **kwargs
        )

        return inputs

    def __call__(self, sentences):
        ids, missing_sentences = [], []
        for i, s in enumerate(sentences):
            if s not in self.embeddings_cache and s not in missing_sentences:
                ids.append(i)
                missing_sentences.append(s)

        if len(missing_sentences) > 0:
            missing_sentences_tokenization = self._parse_and_tokenize(missing_sentences, padding='max_length',
                                                                      truncation=True, max_length=self.max_length)
            missing_sentences_embeddings = self._forward(missing_sentences_tokenization)
        missing_index = 0
        embeddings = []
        for i, s in enumerate(sentences):
            if i in ids:
                sentence_embeddings = missing_sentences_embeddings[missing_index]
                missing_index += 1
                self.embeddings_cache[s] = sentence_embeddings
                embeddings.append(sentence_embeddings)
            else:
                embeddings.append(self.embeddings_cache[s])
        if self.word_dim:
            return np.array(embeddings)[:, :, :self.word_dim]
        else:
            return np.array(embeddings)