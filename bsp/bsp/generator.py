from typing import List
import torch
import time

def _run_and_timing(fn):
    torch.cuda.synchronize()
    start_t = time.time()
    ret = fn()
    torch.cuda.synchronize()
    dur = time.time() - start_t
    return ret, dur

def contains_subarray(filtered_b, subarray):
    sub_len = len(subarray)
    for i in range(len(filtered_b) - sub_len + 1):
        if filtered_b[i:i+sub_len] == subarray:
            return True
    return False

def count_until_eos(tokens, eos_id):
    eos_indices = (tokens == eos_id).nonzero()
    if eos_indices.nelement() == 0:
        return len(tokens)
    first_eos_index = eos_indices[0].item()
    return first_eos_index + 1

def early_stop(generated_tokens, attention_mask, input_ids, stopping_ids, spec_step, eos_id):
    for b in range(input_ids.shape[0]):
        generated_tokens_b = generated_tokens[b]
        attention_mask_b = attention_mask[b]
        filtered_b = generated_tokens_b[attention_mask_b.bool()][-spec_step-4:].tolist()
        for sub in stopping_ids:
            if contains_subarray(filtered_b, sub):
                input_ids[b][-1] = eos_id
                attention_mask[b][-1] = 1
                generated_tokens[b][-1] = eos_id
    return generated_tokens, attention_mask, input_ids
        

class SpeculativeGenerationModel:
    def __init__(self, model, assist_model, tokenizer, specualtive_step=1, device='cuda'):
        self.model = model
        self.assist_model = assist_model
        self.tokenizer = tokenizer
        self.target_device = model.device
        self.assist_defice = assist_model.device

        self.specualtive_step = 1 if specualtive_step is None else specualtive_step

        # stats
        self.pos_correct = torch.zeros([self.specualtive_step], device=self.target_device)
        self.pos_cnt = 0

        self.time_speculate = 0
        self.time_verify = 0
        self.verify_calls = 0

    def _speculative(self, input_ids, attention_mask, kv_cache, speculate_step):
        batch_size = input_ids.shape[0]
        generated_tokens = [[] for _ in range(batch_size)]
        for i in range(speculate_step):
            ret = self.assist_model(input_ids,
                                    attention_mask=attention_mask, 
                                    use_cache=True, 
                                    past_key_values=kv_cache)
            input_ids = torch.argmax(ret.logits[:, -1:], axis=2)

            for b in range(batch_size):
                generated_tokens[b].append(input_ids[b, 0])

            attention_mask = self._extend_mask(attention_mask) 
            kv_cache = ret.past_key_values
            torch.cuda.empty_cache()
        return generated_tokens, attention_mask, kv_cache
    
    def _last_pos_logits(self, logits, mask):
        last_pos = torch.sum(mask, axis=1) - 1
        return logits[torch.arange(logits.shape[0]), last_pos]
    
    def _extend_mask(self, mask):
        return torch.cat([mask, torch.ones([mask.shape[0], 1], device=mask.device, dtype=torch.int32)], axis=1)

    @torch.inference_mode()
    def generate(self, prompts:List[str], num_out:int, collect_stats=False, speculative_step=None, stopping_ids=[]):
        speculative_step = self.specualtive_step if speculative_step is None else speculative_step
        # self.tokenizer.padding_side='right'
        token_seqs = self.tokenizer.batch_encode_plus(prompts, padding=True, return_tensors="pt").to(self.target_device)
        batch_size = len(prompts)
        assist_kv_cache = None
        input_ids = token_seqs['input_ids']
        unified_len = input_ids.shape[1]
        attention_mask = input_attention_mask = token_seqs['attention_mask']

        # prefill initial input attentions with target model; generate the first next token
        ret, t_prefill = _run_and_timing(lambda: self.model(input_ids, attention_mask=input_attention_mask, use_cache=True))
        self.time_verify += t_prefill
        self.verify_calls += 1
        first_token = torch.argmax(self._last_pos_logits(ret.logits, attention_mask), axis=1).unsqueeze(1) 
        attention_mask = self._extend_mask(attention_mask)
        input_ids = torch.cat([input_ids, first_token], axis=1)
        kv_cache = ret.past_key_values
        generated_tokens = input_ids
        valid_lens = torch.ones(batch_size, device=self.target_device)
        step_count = 0
        torch.cuda.empty_cache()

        # stats
        while True:
            # Generate predictions with draft(assist) model
            (speculated_tokens, attention_mask, assist_kv_cache), t_spec = _run_and_timing(lambda: self._speculative(input_ids, attention_mask, assist_kv_cache, speculative_step))
            self.time_speculate += t_spec
            speculated_tokens = torch.tensor(speculated_tokens, device=self.target_device, dtype=torch.int64)   # [batch, gamma]
            verify_inputs = torch.cat([first_token, speculated_tokens], axis=1)
            
            # Verify with target model, generating a "correct" tensor with the correct token values
            ret, t_verify = _run_and_timing(lambda: self.model(verify_inputs, attention_mask=attention_mask, use_cache=True, past_key_values=kv_cache))
            self.time_verify += t_verify
            self.verify_calls += 1
            logits = ret.logits
            kv_cache = ret.past_key_values
            correct = logits[:, :-1].argmax(dim=2)

            # Mask wrong predictions and append full speculated tokens
            check_mask = torch.cumsum(correct == speculated_tokens, 1) == torch.arange(1, speculative_step + 1, device=self.target_device)
            correct_len = torch.sum(check_mask, axis=1)   # Count how many 1's there are
            first_token = torch.argmax(logits[torch.arange(logits.shape[0]), correct_len], axis=1).unsqueeze(1)
            input_ids = torch.concat([speculated_tokens[:, -1:], first_token], axis=1)  # [batch, 2] (only need the last two tokens because the previous ones are saved in cache)
            attention_mask[:, -speculative_step:] = check_mask
            attention_mask = self._extend_mask(attention_mask)   # [batch, full_length]
            generated_tokens = torch.cat([generated_tokens, speculated_tokens, first_token], axis=1)  # [batch, full_length]
            torch.cuda.empty_cache()
            
            # # Check for early stop. If there is, mark a eos_token_id.
            step_count += 1
            if speculative_step>0 and len(stopping_ids)>0 and step_count>=2:
                generated_tokens, attention_mask, input_ids = early_stop(generated_tokens, attention_mask, input_ids, stopping_ids, speculative_step, self.tokenizer.eos_token_id)
            
            # update stats
            if collect_stats:
                not_ended = (valid_lens < num_out).unsqueeze(1)
                self.pos_correct += (check_mask * not_ended).sum(dim=0)
                self.pos_cnt += not_ended.sum()

            # Stop generating when meeting stopping logits or max num tokens generated
            valid_lens += correct_len + 1
            if (generated_tokens[:,unified_len:]==self.tokenizer.eos_token_id).any(dim=1).all() or torch.all(valid_lens >= num_out):
                break
            
        # Collect returned string
        ret = []
        valid_token_count = 0
        attention_mask = attention_mask[:, unified_len:]
        generated_tokens = generated_tokens[:, unified_len:]
        for b in range(batch_size):
            valid_token = torch.nonzero(attention_mask[b], as_tuple=True)[0]
            tokens = generated_tokens[b][valid_token]
            valid_token_num = count_until_eos(tokens, self.tokenizer.eos_token_id)
            valid_token_count += valid_token_num
            tokens = tokens[:valid_token_num]
            answer_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            ret.append(answer_text)
        
        return ret, valid_token_count

    def get_stats(self):
        return self.pos_correct / self.pos_cnt, self.time_speculate, self.time_verify, self.verify_calls

    def reset_stats(self):
        self.pos_correct = 0
        self.pos_cnt = 0
        self.time_speculate = 0
        self.time_verify = 0
        self.verify_calls = 0

