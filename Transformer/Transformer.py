from Transformer_Components import Encoder, Decoder
from EmbeddingModel import EmbeddingModel

class Transformer():

    def __init__(self, hidden_dim, feedforward_dim, num_heads, num_layers,
                 max_decoding_length, vocab_size, padding_index, bos_index,
                 dropout_probability):
        embedding = EmbeddingModel(vocab_size, embedding_dim = hidden_dim, padding_index = padding_index)
        self.encoder = Encoder(hidden_dim, feedforward_dim, num_heads, num_layers, dropout_probability)
        self.decoder = Decoder(hidden_dim, feedforward_dim, num_heads, num_layers, dropout_probability)

        self.padding_index = padding_index
        self.bos_index = bos_index
        self.max_decoding_length = max_decoding_length
        self.hidden_dim = hidden_dim