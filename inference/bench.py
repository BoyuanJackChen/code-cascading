import sys
import time
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
model = AutoModelForCausalLM.from_pretrained(sys.argv[1], torch_dtype=torch.float16, device_map="auto")
model.eval()

print(tokenizer.bos_token_id)

# Generate some text
start = time.time()
with torch.no_grad():
    out = model.generate(
        input_ids=torch.tensor([[tokenizer.bos_token_id]]).to('cuda'),
        # max_new_tokens=500,
        # num_beams = 4,
        # num_beam_groups = 4,
        # diversity_penalty = 0.3,
        max_length=512,
        min_length=512,
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
end = time.time()
assert len(out[0]) == 512
print(f"Generation with {sys.argv[1]} took {end - start} seconds to generate 512 tokens, or {512 / (end - start)} tokens per second")
