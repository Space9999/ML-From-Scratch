import numpy as np
import math
from utils.Layers import Dense, Activation, Dropout, LayerNormalization
import utils.Activations_Functions as Activations_Functions
import utils.Base_Neural_Network as NN
import utils.Loss_Functions as Loss_Functions

# Reference: https://brandonrohrer.com/transformers.html
class MultiHeadAttention():

    def __init__(self, hidden_dim, num_heads):

        # Should be a integer
        self.qkv_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # The 3 comes from the concatentation of the query, key, and value matrices
        self.qkv_projection = Dense(input_size = hidden_dim, n_units = 3 * num_heads * self.qkv_dim, have_bias = False)

        # For cross attention only
        self.query_projection = Dense(input_size = hidden_dim, n_units = num_heads * self.qkv_dim, have_bias = False)
        self.key_value_projection = Dense(input_size = hidden_dim, n_units = 2 * num_heads * self.qkv_dim, have_bias = False)

        self.output_projection = Dense(input_size = num_heads * self.qkv_dim, n_units = hidden_dim, have_bias = False)

    def self_attention_projection(self, x):
        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv_projection.forward_pass(input = x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.qkv_dim)
        query, key, value = np.array_split(qkv, 3, axis = -1)

        return query, key, value

    def cross_attention_projection(self, encoder_hidden_states, decoder_hidden_states):
        batch_size, source_sequence_length, _ = encoder_hidden_states.shape
        _, target_sequence_length, _ = decoder_hidden_states.shape

        query = self.query_projection.forward_pass(input = decoder_hidden_states)
        query = query.reshape(batch_size, target_sequence_length, self.num_heads, self.qkv_dim)

        key_value = self.key_value_projection.forward_pass(input = encoder_hidden_states)
        key_value = key_value.reshape(batch_size, source_sequence_length, self.num_heads, 2 * self.qkv_dim)
        key, value = np.array_split(key_value, 2, axis = -1)

        return query, key, value

    def mask_logits(self, logits, source_padding_mask = None, future_mask = None):
        masked_logits = logits

        if source_padding_mask is not None:
            masked_logits = np.ma.MaskedArray(masked_logits, source_padding_mask[:, np.newaxis, np.newaxis, :] == 0)
            # e^-inf is treated as 0 in python so, this disregards padding tokens from softmax
            masked_logits = masked_logits.filled(fill_value = float("-inf"))

        if future_mask is not None:
            masked_logits = np.ma.MaskedArray(masked_logits, future_mask == 0)
            masked_logits = masked_logits.filled(fill_value = float("-inf"))

        return masked_logits

    def scaled_dot_product(self, query, key, value, source_padding_mask = None, future_mask = None):

        # Swapping the last two axes ensures that the matrix multiplication works even if query and key positions are not of same size
        attention_logits = np.dot(query, np.transpose(key, -2, -1))
        attention_logits = attention_logits / np.sqrt(query.shape[-1])

        if source_padding_mask is not None or future_mask is not None:
            attention_logits = self.mask_logits(attention_logits, source_padding_mask, future_mask)

        attention = Activations_Functions.softmax(attention_logits)
        attention_value = np.dot(attention, value)

        return attention_value

    def forward_pass(self, input, encoder_hidden_states = None, source_padding_mask = None, future_mask = None):
        batch_size, sequence_length, hidden_dim = input.size()

        # "input" variable should be the decoder hidden states
        if encoder_hidden_states is None:
            query, key, value = self.self_attention_projection(input)
        else:
            query, key, value = self.cross_attention_projection(encoder_hidden_states, input)

        # Swapped dimesions to make matrix multiplication work for scaled dot product
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        values = self.scaled_dot_product(query, key, value, source_padding_mask, future_mask)
        values = values.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, hidden_dim)

        output = self.output_projection.forward_pass(values)
        return output

# Specific reference for math: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
class PositionalEncoding():

    # All calculations are done in init method for precomputation
    def __init__(self, hidden_dim, max_length = 5000):
        self.positional_embedding = np.zeros((max_length, hidden_dim))
        position = np.arange(0, max_length, dtype = float)
        position = position[:, np.newaxis]

        # This is mathematically equivalent to the formula shown in the reference
        # Natural log and exponent is more efficient than direct division for vector embeddings
        division_term = np.exp(np.arange(0, max_length, 2, dtype = float) * (-math.log(10000.0) / hidden_dim))
        self.positional_embedding[:, 0::2] = np.sin(position * division_term)
        self.positional_embedding[:, 1::2] = np.cos(position * division_term)

        self.positional_embedding = self.positional_embedding[np.newaxis, :]

    def forward_pass(self, embeddings):
        # Adds positional embeddings up to the sequence size 
        return embeddings + self.positional_embedding[:, : np.size(input, axis = 1)]

class EncoderBlock():

    def __init__(self, hidden_dim, feedforward_dim, num_heads, dropout_probability):
        self.mha = MultiHeadAttention(hidden_dim, num_heads)

        self.feedforward = NN.Base_Neural_Network(loss = Loss_Functions.categorical_cross_entropy)
        self.feedforward.add(Dense(input_size = hidden_dim, n_units = feedforward_dim, have_bias = False))
        self.feedforward.add(Activation("relu"))
        self.feedforward.add(Dense(input_size = feedforward_dim, n_units = hidden_dim, have_bias = False))

        self.dropout1 = Dropout(dropout_probability)
        self.dropout2 = Dropout(dropout_probability)
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()

    def forward_pass(self, input, source_padding_mask):
        block_output = self.dropout1.forward_pass(self.mha.forward_pass(input, source_padding_mask = source_padding_mask))
        block_output_concat = self.layer_norm1.forward_pass(input + block_output)

        block_output2 = self.dropout2.forward_pass(block_output_concat)
        block_output_concat2 = self.layer_norm2.forward_pass(block_output_concat + block_output2)

        return block_output_concat2

class DecoderBlock():

    def __init__(self, hidden_dim, feedforward_dim, num_heads, dropout_probability):
        self.cross_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.self_mha = MultiHeadAttention(hidden_dim, num_heads)

        self.feedforward = NN.Base_Neural_Network(loss = Loss_Functions.categorical_cross_entropy)
        self.feedforward.add(Dense(input_size = hidden_dim, n_units = feedforward_dim, have_bias = False))
        self.feedforward.add(Activation("relu"))
        self.feedforward.add(Dense(input_size = feedforward_dim, n_units = hidden_dim, have_bias = False))

        self.dropout1 = Dropout(dropout_probability)
        self.dropout2 = Dropout(dropout_probability)
        self.dropout3 = Dropout(dropout_probability)
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()
        self.layer_norm3 = LayerNormalization()

    def forward_pass(self, input, encoder_hidden_states, source_padding_mask, future_mask):

        block_output = self.dropout1(self.self_mha.forward_pass(input, future_mask))
        block_output_concat = self.layer_norm1.forward_pass(input + block_output)

        block_output2 = self.dropout2(self.cross_mha.forward_pass(block_output_concat, encoder_hidden_states, source_padding_mask))
        block_output_concat2 = self.layer_norm2.forward_pass(block_output_concat + block_output2)

        block_output3 = self.dropout3(self.feedforward.forward_pass(block_output_concat2))
        block_output_concat3 = self.layer_norm2.forward_pass(block_output_concat2 + block_output3)
        
        return block_output_concat3

class Encoder():

    def __init__(self, hidden_dim, feedforward_dim, embedding, num_heads, num_blocks, dropout_probability):
        self.embedding = embedding
        self.hidden_dim = hidden_dim
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.dropout = Dropout(dropout_probability)
        self.encoder_blocks = []
        for _ in range(num_blocks):
            self.encoder_blocks.append(EncoderBlock(hidden_dim, feedforward_dim, num_heads, dropout_probability))
        
    def forward_pass(self, input_ids, source_padding_mask):
        output = self.embedding.getEmbedding(input_ids) * math.sqrt(self.hidden_dim)
        output = self.positional_encoding.forward_pass(output)
        output = self.dropout.forward_pass(output)

        for encoder_block in self.encoder_blocks:
            output = encoder_block.forward_pass(output, source_padding_mask)

        return output

class Decoder():
    def __init__(self, hidden_dim, feedforward_dim, embedding, vocab_size, num_heads, num_blocks, dropout_probability):
        self.embedding = embedding
        self.hidden_dim = hidden_dim
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.dropout = Dropout(dropout_probability)
        self.decoder_blocks = []
        for _ in range(num_blocks):
            self.decoder_blocks.append(DecoderBlock(hidden_dim, feedforward_dim, num_heads, dropout_probability))

        self.output_layer = Dense(hidden_dim, vocab_size, have_bias = False)
            
    def forward_pass(self, input_tokens, encoder_hidden_states, source_padding_mask, future_mask):
        output = self.embedding.getEmbedding(input_tokens) * math.sqrt(self.hidden_dim)
        output = self.positional_encoding.forward_pass(output)
        output = self.dropout.forward_pass(output)

        for decoder_block in self.decoder_blocks:
            output = decoder_block.forward_pass(output, encoder_hidden_states, source_padding_mask, future_mask)

        output = self.output_layer.forward_pass(output)
        return output








            

        

        

        


