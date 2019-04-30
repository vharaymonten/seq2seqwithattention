# Seq2Seq Model with Attention Mechanism


## Usage

### Training
```
python3 train.py -v 1 --ckpt_prefix checkpoints -epoch 1 -src_dataset dataset/english.txt -tgt_dataset dataset/french.txt -hparams hyperparameters.json
```
### Testing 
```
python3 infer.py --ckpt_prefix checkpoints --hparams hyperparmeters.json
```

## Prerequisites

What things you need to install 

```
numpy
dill
tensorflow 1.13.1
```

## References 

Refererences used in this project
<br>
<ul>
	<li><a href="https://gist.github.com/ilblackdragon/c92066d9d38b236a21d5a7b729a10f12"> seq2seq.py </a></li>
	<li><a href="https://github.com/tensorflow/nmt"> Tensorflow nmt</a></li>
</ul>
