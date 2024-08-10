import argparse
import _settings
import logging
import os
import sys
import openai
import replicate
import json
from tqdm import tqdm
from prompts import experiment_prompts as exp_prompts
from prompts import zero_shot
from .post_process import extract_num

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import requests
from time import sleep

# EXP_NAME = "9_4_claude_existent"

# API keys
openai.api_key = _settings.openai_key
replicate_api_key = os.environ.get("REPLICATE_API_KEY") # for Vicuna-13B
bard_key = os.environ.get("GOOGLE_API_KEY")

# verbalized check prompt
# task_pre_response_prompt = exp_prompts.task_pre_response_prompt
# task_post_response_prompt = exp_prompts.task_post_response_prompt
# knowledge_pre_response_prompt = exp_prompts.knowledge_pre_response_prompt
# knowledge_post_response_prompt = exp_prompts.knowledge_post_response_prompt

task_pre_response_prompt = zero_shot.pre_response_prompt
task_post_response_prompt = zero_shot.post_response_prompt
task_mid_response_prompt = zero_shot.mid_response_prompt
knowledge_pre_response_prompt = zero_shot.honest_pre_response_prompt
knowledge_post_response_prompt = zero_shot.honest_post_response_prompt
def get_prompt(model,prompt_type,dataset_name,line) -> str:
    """
    model: LLMs used
    query_type: "pre" or "post" 
    task_type : 'task' or 'knowledge'
    line: the line["prompt"] from the json file
    
    return:
    the formatted prompt string
    """
    if prompt_type == "pre":
        
        if dataset_name == "task":
            return task_pre_response_prompt.format(line)
        elif dataset_name == 'knowledge':
            return knowledge_pre_response_prompt.format(line)
        else: # for debug
            return task_pre_response_prompt.format(line)
        
    elif prompt_type == "post":
        
        if dataset_name == "task":
            return task_post_response_prompt.format(line)
        elif dataset_name == 'knowledge':
            return knowledge_post_response_prompt.prompt_baseline.format(line)
        else: # for debug
            return task_post_response_prompt.format(line)
        
    elif prompt_type == "mid":
        
         return task_post_response_prompt.format(line)
            
        


def get_existing_samples_in_output_file(path_out) -> set:
    existing_samples_in_output_file = set()
    if not os.path.exists(path_out):
        # if the output file does not exist, then return an empty set
        return existing_samples_in_output_file
        
    with open(path_out, 'r') as f:
        lines_output = f.readlines()
        for line in lines_output:
            data = json.loads(line)
            if data['prompt'] not in existing_samples_in_output_file:
                existing_samples_in_output_file.add(data['prompt'])
    return existing_samples_in_output_file
    

# Error callback function
def log_retry_error(retry_state):  
    print(f"Retrying due to error: {retry_state.outcome.exception()}")  


# OpenAI api: ChatGPT (gpt3.5) / GPT-4
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(1), retry_error_callback=log_retry_error)
def run_GPT_chat(path_in, path_out, prompt_type, dataset_name, model="gpt-3.5-turbo"):
    sequences = []
    num_generation = 3
    with open(path_in, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = json.loads(line)
            prompt = get_prompt(model,prompt_type,dataset_name, line["prompt"])

            response = openai.ChatCompletion.create(
                model=model,
                n=num_generation,
                messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            model_outputs = [response['choices'][i]['message']['content'] for i in range(num_generation)]
            
            prob_outputs = [extract_num(model_outputs[i]) for i in range(num_generation)]
            
            generation = [{'response':model_outputs[i],'probability':prob_outputs[i]} for i in range(num_generation)]
            sequence_dict = {
                        'prompt': line["prompt"],
                        'prompt_type': prompt_type,
                        'generation': generation,
                    }
            sequences.append(sequence_dict)
    # write the json file
    with open(path_out, 'w') as f:
            json.dump(sequences, f)




    
# Vicuna-13B api
# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), retry_error_callback=log_retry_error)
def run_Vicuna13B(path_in, path_out, task_type):
    existing_samples_in_output_file = get_existing_samples_in_output_file(path_out)
    with open(path_in, 'r') as f:
        lines = f.readlines()
        # lines = lines[START:END]
        for line in tqdm(lines):
            line = json.loads(line)
            
            if MODE == "existent":
                if line["label"] == 1 or (line["prompt"] in existing_samples_in_output_file):
                    continue
            elif MODE == "nonexistent":
                if line["label"] == 0 or (line["prompt"] in existing_samples_in_output_file):
                    continue
                
            replicate_client = replicate.Client(api_token=replicate_api_key)
            prompt = get_prompt(task_type, line["prompt"])
            input_params = {
                "prompt": prompt,
                "temperature": 0.1,
                "max_length": MAX_OUT_TOKENS,
                "top_p": 1,
                "repitition_penalty": 1,
            }
            output = replicate_client.run(
                "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
                input=input_params,
            )

            # The predict method returns an iterator (output), res is a string joined from that iterator
            res = "".join(output)

            # # put res into the json file
            line["vicuna-13b"] = res
            # write the json file
            with open(path_out, 'a') as f:
                json.dump(line, f)
                f.write('\n')
                
                
# Llama-2 api - replicate
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15), retry_error_callback=log_retry_error)
def run_Llama2(path_in, path_out, task_type):
    existing_samples_in_output_file = get_existing_samples_in_output_file(path_out)
    with open(path_in, 'r') as f:
        lines = f.readlines()
        # lines = lines[START:END]
        for line in tqdm(lines):
            line = json.loads(line)

            if MODE == "existent":
                if line["label"] == 1 or (line["prompt"] in existing_samples_in_output_file):
                    continue
            elif MODE == "nonexistent":
                if line["label"] == 0 or (line["prompt"] in existing_samples_in_output_file):
                    continue
                
            replicate_client = replicate.Client(api_token=replicate_api_key)
            prompt = get_prompt(task_type, line["prompt"])
            input_params = {
                "prompt": prompt,
                "system_prompt": "Respond in the precise format requested by the user. Do not acknowledge requests with 'sure' or in any other way besides going straight to the answer.",
                "temperature": 0.1,
                "max_new_tokens": MAX_OUT_TOKENS,
                "min_new_tokens": 2,
                "top_p": 1,
            }
            output = replicate_client.run(
                "a16z-infra/llama-2-13b-chat:d5da4236b006f967ceb7da037be9cfc3924b20d21fed88e1e94f19d56e2d3111",
                input=input_params,
            )

            # The predict method returns an iterator (output), res is a string joined from that iterator
            res = "".join(output)

            # # put res into the json file
            line["llama-2-13b"] = res
            # write the json file
            with open(path_out, 'a') as f:
                json.dump(line, f)
                f.write('\n')
                
                
# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15), retry_error_callback=log_retry_error)      
def run_llama2_blender(path_in, path_out, task_type, model, url):
    sys_msg="You are a helpful assistant."
    alt_sys_msg = "Answer the given question in no more than one sentence. Keep your answer short and concise."
    existing_samples_in_output_file = get_existing_samples_in_output_file(path_out)
    with open(path_in, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = json.loads(line)
            if MODE == "existent":
                if line["label"] == 1 or (line["prompt"] in existing_samples_in_output_file):
                    continue
            elif MODE == "nonexistent":
                if line["label"] == 0 or (line["prompt"] in existing_samples_in_output_file):
                    continue
            prompt = get_prompt(task_type, line["prompt"])

            headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' 
            }

            data = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': sys_msg}, 
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0,
                'max_tokens': MAX_OUT_TOKENS
            }
            
            response = requests.post(url, headers=headers, json=data)       
            res = response.json()
            # print(res)
            # break     
            res = response.json()['choices'][0]['message']['content']

            # put res into the json file
            line[model] = res
            
            # write the json file
            with open(path_out, 'a') as f:
                json.dump(line, f)
                f.write('\n')
                

def run_llama_through_openai_api(path_in, path_out, task_type, model, url):
    openai.api_key = "EMPTY"
    openai.api_base = url
    
    print(openai.api_base)
    
    existing_samples_in_output_file = get_existing_samples_in_output_file(path_out)
    
    with open(path_in, 'r') as f:
        lines = f.readlines()
        # lines = lines[START:END]
        for line in tqdm(lines):
            line = json.loads(line)
            
            if MODE == "existent":
                if line["label"] == 1 or (line["prompt"] in existing_samples_in_output_file):
                    continue
            elif MODE == "nonexistent":
                if line["label"] == 0 or (line["prompt"] in existing_samples_in_output_file):
                    continue
           
            
            prompt = get_prompt(task_type, line["prompt"])
            
            
            if 'chat' in model:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=MAX_OUT_TOKENS,
                )
                res = completion['choices'][0]["message"]["content"]
            else:
                completion = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=MAX_OUT_TOKENS,
                    # stop
                    stop=['Question', '\n\n', '\nAnswer']
                )
                res = completion['choices'][0]["text"]
            
            # put res into the json file
            line[model] = res
            # write the json file
            with open(path_out, 'a') as f:
                json.dump(line, f)
                f.write('\n')
                
                
# run Claude
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15), retry_error_callback=log_retry_error)
def run_claude(path_in, path_out, task_type):
    
    existing_samples_in_output_file = get_existing_samples_in_output_file(path_out)
    with open(path_in, 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        line = json.loads(line)
        
        if MODE == "existent":
            if line["label"] == 1 or (line["prompt"] in existing_samples_in_output_file):
                continue
        elif MODE == "nonexistent":
            if line["label"] == 0 or (line["prompt"] in existing_samples_in_output_file):
                continue
            
        prompt = get_prompt(task_type, line["prompt"])
        
        anthropic = Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

        completion = anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=128,
            prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
        )
        
        # put res into the json file
        line['claude-2'] = completion.completion
        
        # write the json file
        with open(path_out, 'a') as f:
            json.dump(line, f)
            f.write('\n')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=['task', 'knowledge','test'], required=True)
    parser.add_argument('--prompt', type=str, choices=['pre', 'post','mid'], required=True)
    parser.add_argument('--model', type=str, choices=['gpt-3.5-turbo','gpt-4' 'palm2', 'llama2','llama2-7b', 'llama2-13b-completion', 'llama2-70b', 'llama2-7b-chat', 'llama2-7b-completion', 'llama2-70b-chat', 'llama2-70b-completion', 'claude2'], required=True)
    
    return parser.parse_args()

# main function
if __name__ == "__main__":
    
    args = vars(parse_args())
    prompt_type = args['prompt']
    model = args['model']
    dataset_name = args['data']
    
    test_in = os.path.join(_settings.DATA_FOLDER,f'{dataset_name}.json')
    
    test_out_folder= os.path.join(_settings.GENERATION_FOLDER,f'{dataset_name}')
    os.makedirs(test_out_folder, exist_ok=True)
    test_out = os.path.join(test_out_folder,f'{dataset_name}-{model}-{prompt_type}.json')
        

    # run the models
    if model == 'gpt-3.5-turbo':
        print("Running gpt-3.5-turbo-0125...")
        run_GPT_chat(test_in, test_out, prompt_type, dataset_name, model="gpt-3.5-turbo-0125")
    elif model == 'gpt-4-turbo':
        print("Running gpt-4-0125-preview...")
        run_GPT_chat(NQ_path_in, NQ_path_out, task_type=task, model="gpt-4-0125-preview")
    elif model == 'gpt-4':
        print("Running gpt-4-0613")
        run_GPT_chat(NQ_path_in, NQ_path_out, task_type=task, model="gpt-4-0613")
    elif model =='llama2' in MODEL: # llama 2 family
        run_llama_through_openai_api(NQ_path_in, NQ_path_out, task_type="NQ", model="Llama-2-13b-chat-hf", url=URL)
    elif MODEL == 'claude2':
        print("Running claude-2...")
        run_claude(NQ_path_in, NQ_path_out, task_type=task)