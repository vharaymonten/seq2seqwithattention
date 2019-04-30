from basemodel import BaseModel
import tensorflow as tf 
class Decoder(BaseModel):
    def __init__(self, mode, hparams, memory, enc_final_state, mem_seq):
        super().__init__(mode, hparams, True)
        self.mode = mode
        self.vocab_size = hparams['tgt_vocab_size']
        self._build_model(hparams, memory, enc_final_state, mem_seq)
    def _build_loss(self, lr, logits, clip=(-1.25, 1.25)):
        weights = tf.cast(tf.not_equal(self.inputs_[:, :-1], 1), tf.dtypes.float32)      
        self.cost = tf.contrib.seq2seq.sequence_loss(logits,self.outputs_, weights)
        
        if self.mode == "train":
            optimizer = tf.train.AdamOptimizer(lr)

            gradients = optimizer.compute_gradients(self.cost)

            #prevent exploding or vanishing gradients
            if clip != None:
                capped_gradients = [(tf.clip_by_value(grad, clip[0], clip[1]), var) for grad, var in gradients if grad is not None]

            self.opt = optimizer.apply_gradients(capped_gradients)
        
        
    def _build_model(self, hparams, memory, enc_final_state, mem_seq):
        '''
        self.decoder_embed = tf.contrib.layers.embed_sequence(self.inputs_,embed_dim=hparams['embed_dim'], 
                                                              vocab_size=hparams['vocab_size'],
                                                              scope='decoder_embedding')
        
        '''
        
        self.embed_weights = tf.Variable(tf.random.uniform((hparams['tgt_vocab_size'], hparams['tgt_embed_dim']),-1,1))
       
        self.decoder_embed = tf.nn.embedding_lookup(self.embed_weights, self.inputs_)
       
        decoder_cells = self._generate_rnn_cell(hparams['cell'], hparams['num_units'], hparams['rnn_layers'], hparams['keep_prob'])

        self.seq_length = self._sequence_length(self.inputs_)
        
        if self.mode == "train" or self.mode == "eval":
            self.helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_embed, self.seq_length, time_major=False)
        elif self.mode == "infer":
            self.helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embed_weights, tf.zeros([self.batch_size],dtype=tf.dtypes.int32), 1)
        
        self.attn_mech = tf.contrib.seq2seq.BahdanauAttention(hparams['query_units'], memory, mem_seq)

        self.attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cells, self.attn_mech, attention_layer_size=hparams['attn_layer_size'])
       
        #out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, hparams['vocab_size'])
        projection_layer = tf.layers.Dense(hparams['tgt_vocab_size']) 
        
        self.initial_state = self.attn_cell.zero_state(self.batch_size, dtype=tf.float32)
        if hparams['use_enc_final_state']:
            self.initial_state = self.initial_state.clone(cell_state=enc_final_state)
        
        self.decoder_ins = tf.contrib.seq2seq.BasicDecoder(self.attn_cell, helper=self.helper,
                                                           initial_state=self.initial_state, output_layer=projection_layer)
        
        self.outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=self.decoder_ins, output_time_major=self.time_major,
                impute_finished=True, maximum_iterations=2*hparams['dec_max_time_step'])
        
        self.logits = self.outputs[0].rnn_output
        self.sample_id = self.outputs[0].sample_id
        if self.mode != "infer":
            self._build_loss(hparams['lr'], self.logits)
    
    
