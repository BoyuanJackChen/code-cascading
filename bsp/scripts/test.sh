python scripts/benchmark.py \
       --model facebook/opt-6.7b  \
       --assist-model facebook/opt-125m\
       --tokenizer facebook/opt-125m\
       --len-out 128 \
       --speculate-step 0 1 2 3 4 5 6 7 8\
       --batch-size 1 2 4 8 16 32\
       --fp16 \
       --dataset alespalla/chatbot_instruction_prompts \
       --dataset-truncate 3 \
       --warmup 1\
       --collect-stats
       
# This is correct
CUDA_VISIBLE_DEVICES=0,1 python scripts/benchmark.py \
       --model WizardLM/WizardCoder-3B-V1.0 \
       --assist-model WizardLM/WizardCoder-1B-V1.0\
       --tokenizer WizardLM/WizardCoder-1B-V1.0\
       --len-out 1024 \
       --speculate-step 0\
       --batch-size 2\
       --dataset humaneval \
       --dataset-truncate 2 \
       --warmup 0\
       --collect-stats

# This is correct
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py \
       --model WizardLM/WizardCoder-3B-V1.0 \
       --assist-model WizardLM/WizardCoder-1B-V1.0\
       --tokenizer WizardLM/WizardCoder-1B-V1.0\
       --len-out 1024 \
       --speculate-step 4\
       --batch-size 10\
       --dataset apps-intro \
       --dataset-truncate 10 \
       --warmup 0\
       --collect-stats

# This is correct
python scripts/benchmark.py \
       --model WizardLM/WizardCoder-Python-34B-V1.0 \
       --assist-model WizardLM/WizardCoder-Python-7B-V1.0\
       --tokenizer WizardLM/WizardCoder-Python-7B-V1.0\
       --len-out 1024 \
       --speculate-step 4\
       --batch-size 4\
       --dataset humaneval \
       --dataset-truncate 4 \
       --warmup 0\
       --collect-stats

# This is correct
python scripts/benchmark.py \
       --model WizardLM/WizardCoder-Python-34B-V1.0 \
       --assist-model WizardLM/WizardCoder-Python-7B-V1.0\
       --tokenizer WizardLM/WizardCoder-Python-7B-V1.0\
       --len-out 1024 \
       --speculate-step 4\
       --batch-size 12\
       --dataset humaneval \
       --dataset-truncate 60 \
       --warmup 3\
       --collect-stats

python scripts/benchmark.py \
       --model WizardLM/WizardCoder-Python-34B-V1.0 \
       --assist-model WizardLM/WizardCoder-Python-7B-V1.0\
       --tokenizer WizardLM/WizardCoder-Python-7B-V1.0\
       --len-out 1024 \
       --speculate-step 4\
       --batch-size 4\
       --dataset humaneval \
       --dataset-truncate 4 \
       --warmup 0\
       --collect-stats

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/benchmark.py \
       --model Salesforce/codegen-6B-mono \
       --assist-model Salesforce/codegen-350M-mono\
       --tokenizer Salesforce/codegen-350M-mono\
       --len-out 128 \
       --speculate-step 4\
       --batch-size 4\
       --dataset humaneval \
       --dataset-truncate 4 \
       --warmup 0\
       --collect-stats