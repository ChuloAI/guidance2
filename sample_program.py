import os
import glob

from exllama_lib.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama_lib.tokenizer import ExLlamaTokenizer
from exllama_lib.generator import ExLlamaGenerator
import guidance

model_directory =  "./orca_mini_7B-GPTQ/"
tokenizer_path = os.path.join(model_directory, "tokenizer.model")
model_config_path = os.path.join(model_directory, "config.json")
st_pattern = os.path.join(model_directory, "*.safetensors")
model_path = glob.glob(st_pattern)[0]

config = ExLlamaConfig(model_config_path)               # create config from config.json
config.model_path = model_path                          # supply path to model weights file

model = ExLlama(config)                                 # create ExLlama instance and load the weights
tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file

cache = ExLlamaCache(model)                             # create cache for inference
generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator

llm = guidance.llms.ExLlama(model=model, generator=generator, tokenizer=tokenizer, model_config=config)

program = guidance('''Is the man female or male"{{#select 'answer'}}female{{or}}male{{/select}}"''', llm=llm)
out = program()
print(out)
print(out["answer"])


program = guidance('''Is the man female or male? Think carefully and answer this question. {{#select 'answer'}}female{{or}}male{{/select}}. What do you think? "{{gen 'something'}}" ''', llm=llm)
out = program()
print("Gen variable:", out["answer"])
print("GEnerated text: ", out)