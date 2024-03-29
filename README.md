## Tinyllama Bitnet
This repository demonstrates training a small ~84M param BitNet model based on the llama2 architecture.
The entire process including preparing the data, defining the model architecture, and training the model is available in train.py. I wanted to make this process as straight-forward and hackable as possible.
#### Training Data
The script currently uses a 15% subset of openwebtext2 for training. This has been pretokenized at a context length of 256 for ease of testing, but code is also included to tokenize data yourself.
You can replace a couple lines in the script to train on pretty much anything else you want.

## BitNet
The BitLinear definition is copied straight from the released training details [manuscript](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf).
The BitNet architecture is defined by loading a blank Llama2 model using huggingface, and then making the necessary replacements (as per the manuscript):
1. Replace all nn.Linear in attention and SwiGLU with BitLinear
2. Remove RMSNorm before attention and SwiGLU because BitLinear has built-in RMSNorm.
