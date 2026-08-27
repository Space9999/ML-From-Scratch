from Transformer_Components import Encoder, Decoder
from EmbeddingModel import EmbeddingModel
from utils.Loss_Functions import categorical_cross_entropy, categorical_cross_entropy_grad
import numpy as np

class Transformer():

    def __init__(self, hidden_dim, feedforward_dim, num_heads, num_blocks,
                 max_decoding_length, vocab_size, padding_index, bos_index,
                 dropout_probability):
        embedding = EmbeddingModel(vocab_size, embedding_dim = hidden_dim, padding_index = padding_index)
        self.encoder = Encoder(hidden_dim, feedforward_dim, embedding, num_heads, num_blocks, dropout_probability)
        self.decoder = Decoder(hidden_dim, feedforward_dim, embedding, num_heads = num_heads, 
                               num_layers = num_blocks, dropout_probability = dropout_probability)

        self.padding_index = padding_index
        self.bos_index = bos_index
        self.max_decoding_length = max_decoding_length
        self.hidden_dim = hidden_dim

    def train_transformer(self, epochs, batches, masks):
        for epoch in range(epochs):
            # Iterates through rows of batches appended with appropriate masks
            for i, (src_batch, src_mask, tgt_batch, tgt_mask) in enumerate(
                zip(batches["src"], masks["src"], batches["tgt"], masks["tgt"])
            ):
                encoder_output = self.encoder.forward_pass(src_batch, src_mask)
                decoder_output = self.decoder.forward_pass(tgt_batch, encoder_output,
                                                           src_mask, tgt_mask)
                # Last decoder output is meaningless as it does not have a target token
                decoder_output = decoder_output[:, :-1, :]
                # The BOS token should not be included in loss
                tgt_batch = tgt_batch[:, 1:]

                # Ignore padding values in loss and accuracy using mask
                tgt_padding_mask = tgt_batch != self.padding_index

                batch_loss = categorical_cross_entropy(tgt_batch[tgt_padding_mask], decoder_output.transpose(0, 2, 1))

                batch_accuracy = (np.sum(decoder_output.argmax(dim = -1) == tgt_batch)) / tgt_batch.size

                


                



                

                


