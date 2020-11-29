from transformers import FeatureExtractionPipeline
import numpy as np
import pickle
import os


class CachedFeatureExtractionPipeline(FeatureExtractionPipeline):

    def __init__(self, word_dim, max_length, name, *args, pickle_cache=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings_cache = {}
        self.word_dim = word_dim
        self.max_length = max_length
        self.name = name
        self.pickle_cache = pickle_cache
        self.load_cache()
        self.last_size = len(self.embeddings_cache)

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
        self.save_cache()

        if self.word_dim:
            return np.array(embeddings)[:, :, :self.word_dim]
        else:
            return np.array(embeddings)

    def save_cache(self):
        if len(self.embeddings_cache) - self.last_size > 5000:
            self.last_size = len(self.embeddings_cache)
            with open(f'.{self.name}cache', 'wb+') as f:
                pickle.dump(self.embeddings_cache, f)

    def load_cache(self):
        if os.path.exists(f'.{self.name}cache'):
            try:
                with open(f'.{self.name}cache', 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
            except EOFError:
                pass

    def __getstate__(self):
        data = self.__dict__.copy()
        if not self.pickle_cache:
            del data['embeddings_cache']
        return data

    def __setstate__(self, state):
        self.__dict__ = state
        self.load_cache()
        self.last_size = len(self.embeddings_cache)