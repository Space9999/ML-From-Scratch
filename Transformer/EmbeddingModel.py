from utils.Data_Functions import Data_Functions
from utils.Layers import Dense, Activation
import utils.Base_Neural_Network as NN
import utils.Loss_Functions as Loss_Functions
import utils.Optimizers as Optimizers
import numpy as np

# Reference: https://jaketae.github.io/study/word2vec/
class EmbeddingModel():

    def __init__(self, vocab_size, embedding_dim, padding_index = None):
        self.padding_index = padding_index
        
        if (self.padding_index is not None):
            # Padding index should not be included in model training
            self.vocab_size -= 1

        self.embedding_model = NN.Base_Neural_Network(optimizer = Optimizers.Adam(), loss = Loss_Functions.categorical_cross_entropy, loss_grad = Loss_Functions.categorical_cross_entropy_grad)
        self.layer1 = Dense(input_size = vocab_size, n_units = embedding_dim, have_bias = False)
        self.embedding_model.add(self.layer1) # Layer 1 weights contain the appropriate embeddings
        self.embedding_model.add(Dense(input_size = embedding_dim, n_units = vocab_size, have_bias = False))

    def generate_training_data(self, one_hot_encodings, context_window = 2):
        X = []
        y = []

        n_tokens = len(one_hot_encodings)

        for i in range(n_tokens):

            window_indices = np.concat(range(max(0, i - context_window), i), 
                                        range(i + 1, min(n_tokens, i + context_window + 1)))
            for j in window_indices:
                X.append(one_hot_encodings[i])
                y.append(one_hot_encodings[j])

        return np.asarray(X), np.asarray(y)

    def getEmbedding(self, token_sequence):
        one_hot_encodings = Data_Functions.to_categorical(token_sequence, n_col = self.vocab_size)

        X, y = self.generate_training_data(one_hot_encodings)
        self.embedding_model.fit(X, y, epochs = 20, batch_size = 80)

        embedding = self.layer1.get_weight()
        # Add padding embedding as simply zeros
        np.insert(embedding, self.padding_index, np.zeros(self.embedding_dim))

        return embedding



    





        

