import tensorflow as tf
class BaseModel(object):
    def __init__(self, mode, hparams, is_decoder=False):

        '''
        if mode == "train" or mode == "test":
            hparams['batch_size'] = hparams['batch_size']
        else:
            hparams['batch_size'] = tf.placeholder(shape=[None,], dtype=tf.int32)
        
        '''
        if mode == "train" or mode=="eval":
            self.batch_size = hparams['batch_size']
            self.inputs_ = tf.placeholder(shape=(self.batch_size, None), dtype=tf.int32)
            if is_decoder: 
                # Only if the derived model is decoder
                start_tokens = tf.expand_dims(tf.zeros([self.batch_size],dtype=tf.dtypes.int32),1)
                temp_inp_ = tf.concat([start_tokens, self.inputs_], axis=1)
                
                # Swap the variables' name for simplicity
                self.outputs_ = self.inputs_
                self.inputs_ = temp_inp_
        elif mode == "infer":
            self.inputs_ = tf.placeholder(shape=(1, None, ), dtype=tf.int32)
            self.batch_size = 1
            
        self.time_major = hparams['time_major'] 
    
    def _build_model(self, *kw):
        pass
    
    def _sequence_length(self, x):
        
        return tf.reduce_sum(tf.cast(tf.not_equal(x, 1), dtype=tf.int32), 1)

    def eval(self, *kw):
        pass
    
    def inference(self, *kw):
        pass
    
    def train(self, *kw):
        pass
    
    def test(self, *kw):
        pass

    def train_batch(self, *kw):
        pass


    @staticmethod
    def _generate_rnn_cell(cell, num_units, rnn_layers=2, output_keep_prob=0.8, input_keep_prob=1.0,state_keep_prob=1.0):
        '''
        Arguments
        cell (str)                   : either GRU, LSTM, or RNN
        num_units (int)              : number of units per layer
        dropouts (float)             : drop out keep_prob
        '''

        if cell == "GRU":
            cell = tf.nn.rnn_cell.GRUCell
        elif cell == "LSTM":
            cell = tf.nn.rnn_cell.LSTMCell
        else:
            cell = tf.nn.rnn_cell.LSTMCell

        if rnn_layers > 1:    

            cells = [cell(num_units) for i in range(rnn_layers)]
            decoder_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
        else:
            decoder_cells = tf.nn.rnn_cell.GRUCell(num_units)

        decoder_cells = tf.contrib.rnn.DropoutWrapper(decoder_cells, output_keep_prob=output_keep_prob,\
         state_keep_prob=state_keep_prob, input_keep_prob=input_keep_prob)

        return decoder_cells