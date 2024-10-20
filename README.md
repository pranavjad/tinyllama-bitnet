## Tinyllama Bitnet
This repository demonstrates training your own BitNet model based on the llama2 architecture. Unedited, the script will train a ~84M param model on ~1.5B tokens.
#### File structure
**train.py** - the entire training process including preparing the data, defining the model architecture, and training model.

**utils.py** - contains the BitLinear implementation, and convert_to_bitnet function for converting huggingface's LlamaForCausalLM to BitNet.

**inference.py** - run inference with a trained BitNet model.

I wanted to make this process as straight forward and hackable as possible, so all of these scripts are minimal and easily adjustable.
#### Training Data
The script currently uses a 15% subset of openwebtext2 for training. This has been pretokenized at a context length of 256 for ease of testing, but code is also included to tokenize data yourself.
You can replace a couple lines in the script to train on pretty much anything else you want.
#### Dependencies
You'll want to install these packages. The last two are optional and are for logging and HF auth.
- [transformers](https://huggingface.co/docs/transformers/en/installation)
- [datasets](https://huggingface.co/docs/datasets/en/installation)
- [torch](https://pytorch.org/get-started/locally/)
- [wandb](https://docs.wandb.ai/quickstart)
- [huggingface_hub](https://huggingface.co/docs/huggingface_hub/en/installation)

## BitNet
The BitLinear definition is copied straight from the released training details [manuscript](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf).
The BitNet architecture is defined by loading a blank Llama2 model using huggingface, and then making the necessary replacements (as per the manuscript):
1. Replace all nn.Linear in attention and SwiGLU with BitLinear
2. Remove RMSNorm before attention and SwiGLU because BitLinear has built-in RMSNorm.
