from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *

# Load a pretrained llama-based bitnet model
model_path = "<path_to_checkpoint>"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Huggingface thinks it is a LLamaForCausalLM, so we need to convert it to bitnet.
# set inference=True to copy weights and pre-quantize them.
convert_to_bitnet(model, inference=True)

# See the ternary weights
print(model.model.layers[0].self_attn.q_proj.weight)

# Put the model in eval mode so it knows to expect ternary weights
model.eval()

model.to(device="cuda:0")
prompt = "In a shocking turn of events"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generate_ids = model.generate(inputs.input_ids, max_length=100)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

### Uncomment the following lines to test wikitext perplexity using lm-evaluation-harness
### For install instructions see: https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#install
# import lm_eval
# from lm_eval.models.huggingface import HFLM
# eval_model = HFLM(model)
# task_manager = lm_eval.tasks.TaskManager()
# wikitext_results = lm_eval.simple_evaluate(
#     model=eval_model,
#     tasks=["wikitext"],
#     task_manager=task_manager,
#     device="cuda:0",
#     batch_size="auto"
# )
# print(wikitext_results['results'])