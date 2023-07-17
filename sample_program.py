import os
import glob

import guidance

from transformers import LlamaForCausalLM, LlamaTokenizer

model_directory =  "./orca_mini_7B-GPTQ/"
tokenizer = ExLlamaTokenizer.from_pretrained(model_directory)
model = ExLlamaForCausalLM.from_pretrained(model_directory)

llm = guidance.llms.Transformers(model=model, tokenizer=tokenizer)

program = guidance('''Is the man female or male"{{#select 'answer'}}female{{or}}male{{/select}}"''', llm=llm)
out = program()
print(out)
print(out["answer"])


program = guidance('''Is the man female or male? Think carefully and answer this question. {{#select 'answer'}}female{{or}}male{{/select}}. What do you think? "{{gen 'something'}}" ''', llm=llm)
out = program()
print("Gen variable:", out["answer"])
print("GEnerated text: ", out)