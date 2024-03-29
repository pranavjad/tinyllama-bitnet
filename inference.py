from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *

# Load a pretrained BitNet model
model_path = "<path_to_checkpoint>"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Huggingface thinks it is a LLamaForCausalLM, so we need to convert it to bitnet.
# copy_weights=True to pretrained weights get retained when we replace nn.Linear with Bitlinear.
convert_to_bitnet(model, copy_weights=True)
model.to(device="cuda:0")

prompt = "In a shocking turn of events"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generate_ids = model.generate(inputs.input_ids, max_length=100)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]