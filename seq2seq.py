from encoder import *
from decoder import *


class Model(BaseModel):
    def __init__(self, mode, hparams):
        self.hparms = hparams
        
        self._build_model(mode, hparams)
        self.saver = tf.train.Saver()
    
    def _build_model(self, mode, hparams):
        
        self.encoder = Encoder(mode,hparams)
        self.memory, enc_final_state = self.encoder.inference()    
        
        self.decoder = Decoder(mode, hparams, self.memory, enc_final_state, self.encoder.seq_length)

        # copy encoder's and decoder's placeholder for later use
        self.enc_inp_ = self.encoder.inputs_
        if mode == "train" or mode == "eval":
            self.dec_out_ = self.decoder.outputs_
    def inference(self, sess, x):
        return sess.run(self.decoder.sample_id, feed_dict={self.enc_inp_:x})

    def eval(self, sess, X, y):
        loss = sess.run(self.decoder.cost, feed_dict={self.enc_inp_:X, self.dec_out_:y})
        return loss
    
    def train_batch(self, sess, src_inputs, tgt_outputs):
        _, loss = sess.run([self.decoder.opt, self.decoder.cost], feed_dict={self.enc_inp_:src_inputs, 
                           self.dec_out_:tgt_outputs})
        
        return loss
        
    def train(self, sess, iterator, print_nsteps, e=1, verbose=True):
        for step, (X, y) in enumerate(iterator):
            loss = self.train_batch(sess, X, y)
            step += 1 
            if step % print_nsteps == 0:
                 print(" Epoch : {} Step : {} Loss : {} ".format(e, step, loss))
                

        
    
    

    