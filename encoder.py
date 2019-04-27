import tensorflow as tf
from basemodel import BaseModel
class Encoder(BaseModel):
    def __init__(self, mode, hparams):
        super().__init__(mode, hparams)
        self._build_model(hparams)
    def _build_model(self, hparams):
        self.seq_length = self._sequence_length(self.inputs_)


        
        self.encoder_embed = tf.contrib.layers.embed_sequence(self.inputs_, vocab_size=hparams['src_vocab_size'], 
                                           embed_dim=hparams['src_embed_dim'], scope='encoder_embedding')

        encoder_cells =self._generate_rnn_cell("GRU", hparams['num_units'], hparams['rnn_layers'], hparams['keep_prob'])

        self.encoder_cells = tf.contrib.rnn.DropoutWrapper(encoder_cells, output_keep_prob=hparams['keep_prob'])
        self.initial_state = self.encoder_cells.zero_state(self.batch_size, dtype=tf.float32)

        self.outputs, self.final_state = tf.nn.dynamic_rnn(self.encoder_cells, 
                                                                     self.encoder_embed, time_major=self.time_major,
                                                                    initial_state=self.initial_state,dtype=tf.float32)
    def inference(self):
        return self.outputs, self.final_state