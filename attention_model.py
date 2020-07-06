import tensorflow as tf

from attention_graph_encoder import GraphAttentionEncoder
from attention_graph_decoder import GraphAttentionDecoder
from Enviroment import VRPproblem

def set_decode_type(model, decode_type):
    model.set_decode_type(decode_type)
    model.decoder.set_decode_type(decode_type)

class AttentionModel(tf.keras.Model):

    def __init__(self,
                 embedding_dim,
                 n_encode_layers=2,
                 n_heads=8,
                 tanh_clipping=10.
                 ):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.problem = VRPproblem
        self.n_heads = n_heads

        self.embedder = GraphAttentionEncoder(input_dim=self.embedding_dim,
                                              num_heads=self.n_heads,
                                              num_layers=self.n_encode_layers
                                              )

        self.decoder = GraphAttentionDecoder(num_heads=self.n_heads,
                                             output_dim=self.embedding_dim,
                                             tanh_clipping = tanh_clipping,
                                             decode_type = self.decode_type)

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type


    def _calc_log_likelihood(self, _log_p, a):

        # Get log_p corresponding to selected actions
        log_p = tf.gather_nd(_log_p, tf.cast(tf.expand_dims(a, axis=-1), tf.int32), batch_dims=2)

        # Calculate log_likelihood
        return tf.reduce_sum(log_p,1)

    def call(self, input, return_pi=False):
        embeddings, mean_graph_emb = self.embedder(input)

        _log_p, pi = self.decoder(inputs=input, embeddings=embeddings, context_vectors=mean_graph_emb)

        cost = self.problem.get_costs(input, pi)

        ll = self._calc_log_likelihood(_log_p, pi)

        if return_pi:
            return cost, ll, pi

        return cost, ll
