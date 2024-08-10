import getpass
import os
import sys

# __USERNAME = getpass.getuser()

openai_key = "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" # 临时key
#openai_key = "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" # used my own key--wenbo
# openai_key = "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" # used my own key--zihang

#anthropic_key = "sk-5jMA2ZiuV1UUQBy6931c1578029e42F587B56aD507B2314b"
# sk-5jMA2ZiuV1UUQBy6931c1578029e42F587B56aD507B2314b   # from taobao 
# claude-2.1
anthropic_key = "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"
#anthropic_key = "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"

google_api_key = "AIzaSyD7m5OriG32i2IN6154CfzqchHbZKo8ha8"

# AIzaSyD7m5OriG32i2IN6154CfzqchHbZKo8ha8

# old google api key : AIzaSyD0e0miBUIp6pd2tszdzFi-iKR0rh2sMt8

#_BASE_DIR = f'/home/ec2-user/SageMaker/can_benchmark/'

_BASE_DIR = f'/Users/zihang/code/doable_benchmark_opensource/feasible-benchmark'

DATA_FOLDER = os.path.join(_BASE_DIR, 'dataset')
DATA_FOLDER_TASK = os.path.join(DATA_FOLDER,'raw_data','task')
GENERATION_FOLDER = os.path.join(DATA_FOLDER, 'output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)

# After running pipeline/generate.py, update the following paths to the generated files if necessary.
GEN_PATHS = {
    'coqa': {
        'llama-13b': f'{GENERATION_FOLDER}/llama-13b-hf_coqa_10/0.pkl',
        'gpt-3.5-turbo': f'{GENERATION_FOLDER}/gpt-3.5-turbo_coqa_10/0.pkl',
    },
    'trivia': {
        'llama-13b': f'{GENERATION_FOLDER}/llama-13b-hf_triviaqa_10/0.pkl',
        'gpt-3.5-turbo': f'{GENERATION_FOLDER}/gpt-3.5-turbo_triviaqa_10/0.pkl',
    },
    'nq_open':{
        'llama-13b': f'{GENERATION_FOLDER}/llama-13b-hf_nq_open_10/0.pkl',
        'gpt-3.5-turbo': f'{GENERATION_FOLDER}/gpt-3.5-turbo_nq_open_10/0.pkl',
    }
}