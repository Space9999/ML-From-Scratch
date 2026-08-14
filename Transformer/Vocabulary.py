import re

class Vocabulary:
    BOS = "BOS"
    EOS = "EOS"
    PAD = "PAD" # Special tokens

    def __init__(self, sentence_list):
        self.token2index = {self.BOS : 0, self.EOS : 1, self.PAD : 2,}
        self.index2token = {0 : self.BOS, 1 : self.EOS, 2 : self.PAD}

        if sentence_list is None:
            return

        for sentence in sentence_list:
            self.add_vocab(self.tokenize(sentence))

    def add_vocab(self, tokens):
        for token in tokens:
            if token not in self.token2index:
                last_index = len(self.token2index.items())
                self.token2index[token] = last_index
                self.index2token[last_index] = token

    def tokenize(self, sentence, add_special_tokens = True):
        # Match either a sequence of word characters or a sequence of punctuation/symbol characters
        tokens = re.findall(r"\w+|[^\s\w]+", sentence)

        if add_special_tokens:
            tokens = [self.BOS] + tokens + [self.EOS]
        return tokens

    def encode(self, sentence, add_special_tokens = True):
        tokens = self.tokenize(sentence, add_special_tokens)
        return [self.token2index[token] for token in tokens]

    def decode(self, indices):
        return [self.index2token[index] for index in indices]




