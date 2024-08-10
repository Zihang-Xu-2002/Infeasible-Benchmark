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
import numpy as np
import asyncio

import pprint
import google.generativeai as palm

from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import requests
from time import sleep


# EXP_NAME = "9_4_claude_existent"

# API keys
#print(_settings.openai_key)
openai.api_key = _settings.openai_key
google_api_key = _settings.google_api_key
replicate_api_key = os.environ.get("REPLICATE_API_KEY") # for Vicuna-13B


# verbalized check prompt
# task_pre_response_prompt = exp_prompts.task_pre_response_prompt
# task_post_response_prompt = exp_prompts.task_post_response_prompt
# knowledge_pre_response_prompt = exp_prompts.knowledge_pre_response_prompt
# knowledge_post_response_prompt = exp_prompts.knowledge_post_response_prompt

task_pre_response_prompt = zero_shot.pre_response_prompt
task_post_response_prompt = zero_shot.post_response_prompt
task_mid_response_prompt = zero_shot.mid_response_prompt
task_mid_response_prompt_no_step = zero_shot.mid_response_prompt_no_step
task_mix_response_prompt = zero_shot.mix_response_prompt
knowledge_pre_response_prompt = zero_shot.honest_pre_response_prompt
knowledge_post_response_prompt = zero_shot.honest_post_response_prompt
style_transfer_prompt = zero_shot.style_transfer_prompt
style_transfer_prompt_no_example = zero_shot.style_transfer_prompt_no_example
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
        
         return task_mid_response_prompt.format(line)
    elif prompt_type == "mid_no_step":
             return task_mid_response_prompt_no_step.format(line)
    elif prompt_type == "mix":
             return task_mix_response_prompt.format(line)
    
    elif prompt_type == "style_transfer":
        return style_transfer_prompt.format(line)
    
    elif prompt_type == "style_transfer_no_example":
        return style_transfer_prompt_no_example.format(line)
            
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
def run_GPT_chat(path_in, path_out, prompt_type, dataset_name, model="gpt-3.5-turbo", ratio=1.0,mode="withCategory"):
    sequences = []
    num_generation = 5

    with open(path_in, 'r') as f:
            datas = json.load(f)
    
    #从data中随机抽取ratio比例的数据
    if ratio < 1.0:
        datas = np.random.choice(datas, int(len(datas)*ratio), replace=False)
        print(len(datas))

    print("here")

    for line in tqdm(datas):
        retries = 0
        max_retries = 30  # 设定最大重试次数以避免无限循环
        while retries < max_retries:
            prompt = get_prompt(model, prompt_type, dataset_name, line["prompt"])

            try:
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
            except Exception as e:
                print(f"Request timed out, retrying... (Attempt {retries+1}/{max_retries})")
                print(f"Error: {e}")
                retries += 1
                continue
            model_outputs = [response['choices'][i]['message']['content'] for i in range(num_generation)]
                
            prob_outputs = [extract_num(model_outputs[i]) for i in range(num_generation)]

            if 'None' in prob_outputs:  # 检测到None值，准备重试
                retries += 1
                sleep(2)
                print(f"Detected None in probabilities, retrying... (Attempt {retries}/{max_retries})")
                continue  # 跳过当前循环的剩余部分，开始下一次重试

            # 如果没有检测到None，或者已达到最大重试次数，跳出循环
            break

        if retries == max_retries:
            print("Reached max retries, proceeding with current data...")
            # 在这里可以决定如何处理达到最大重试次数的情况，比如使用默认值

        generation = [{'response': model_outputs[i], 'probability': prob_outputs[i]} for i in range(num_generation)]
        if mode == "withCategory":
            sequence_dict = {
                        'prompt': line["prompt"],
                        'prompt_type': prompt_type,
                        'category': line["category"],
                        'generation': generation,
                        }
        else:
            sequence_dict = {
                        'prompt': line["prompt"],
                        'prompt_type': prompt_type,
                        'generation': generation,
                        }

        
        sequences.append(sequence_dict)
    # write the json file
    with open(path_out, 'w') as f:
            json.dump(sequences, f,indent=4)

    # save the confidence and the label(if In_Distribution in dataset name, it is 1, otherwise 0) in a csv file
    with open(f'{path_out[:-5]}.csv', 'w') as f:
        # n generation probability, take the mean value
        if mode == "withCategory":
            f.write('probability,label,variance,category\n')
            for line in sequences:
                prompt = line['prompt']
                prompt_type = line['prompt_type']
                generation = line['generation']
                category = line['category']

                probabilities = [float(gen['probability']) for gen in generation]
                variance = np.var(probabilities) # calculate the variance of the probabilities
                prob = sum([gen['probability'] for gen in generation])/num_generation

                if "undoable" in dataset_name:               
                    f.write(f'{prob},0,{variance},{category}\n')
                else:    
                    f.write(f'{prob},1,{variance},{category}\n')
        else:
            f.write('probability,label,variance\n')
            for line in sequences:
                prompt = line['prompt']
                prompt_type = line['prompt_type']
                generation = line['generation']
                probabilities = [float(gen['probability']) for gen in generation]
                variance = np.var(probabilities) # calculate the variance of the probabilities
                prob = sum([gen['probability'] for gen in generation])/num_generation
                if "undoable" in dataset_name or "mix" in dataset_name:               
                    f.write(f'{prob},0,{variance}\n')
                else:    
                    f.write(f'{prob},1,{variance}\n')



def run_llama_replicate(path_in, path_out, prompt_type, dataset_name, model="gpt-3.5-turbo", ratio=1.0,mode="withCategory"):
    sequences = []
    num_generation = 5
    print("LLAMA-3-70B")

    with open(path_in, 'r') as f:
            datas = json.load(f)
    
    #从data中随机抽取ratio比例的数据
    if ratio < 1.0:
        datas = np.random.choice(datas, int(len(datas)*ratio), replace=False)
        print(len(datas))


    for entry in tqdm(datas):
        prompt = get_prompt(model,prompt_type,dataset_name, entry["prompt"])
        model_outputs = []
        prob_outputs = []
        
        input = {
                    "top_p": 1,
                    "prompt": prompt,
                    "temperature": 0.75,
                    "system_prompt": "You are a helpful  assistant.",
                    "max_new_tokens": 500
                }
        
        for i in range(num_generation):
            retries = 0
            max_retries = 60  
            while retries < max_retries:
                try:
                    output = replicate.run(
                                    "meta/llama-2-70b-chat",
                                    input=input
                                )
                    output = "".join(output)
                    
                    model_outputs.append(output)
                    prob = extract_num(output)
                    if prob == 'None' and retries < max_retries - 1:
                       
                        print(f"Detected None in probabilities, retrying... (Attempt {retries}/{max_retries})")
                        retries += 1
                        time.sleep(2)
                        continue  # 跳过当前循环的剩余部分，开始下一次重试

                    elif prob == 'None' and retries == max_retries - 1:
                        print(f"Reached max retries, proceeding with current data...")
                        prob = 0.5
                        retries += 1
                        prob_outputs.append(prob)
                        print("i-th:",i, "length of prob_outputs:", len(prob_outputs))
                        break
                        
                    prob_outputs.append(prob)

                    break
                except Exception as e:
                    print(f"Request timed out, retrying... (Attempt {retries+1}/{max_retries})")
                    print(f"Error: {e}")
                    time.sleep(5)
                    retries += 1
                    continue
        generation = [{'response': model_outputs[i], 'probability': prob_outputs[i]} for i in range(num_generation)]
        if mode == "withCategory":
            sequence_dict = {
                        'prompt': line["prompt"],
                        'prompt_type': prompt_type,
                        'category': line["category"],
                        'generation': generation,
                        }
        else:
            sequence_dict = {
                        'prompt': line["prompt"],
                        'prompt_type': prompt_type,
                        'generation': generation,
                        }
        
        sequences.append(sequence_dict)
    
    with open(path_out, 'w') as f:
            json.dump(sequences, f,indent=4)

    # save the confidence and the label(if In_Distribution in dataset name, it is 1, otherwise 0) in a csv file
    with open(f'{path_out[:-5]}.csv', 'w') as f:
        # n generation probability, take the mean value
        if mode == "withCategory":
            f.write('probability,label,variance,category\n')
            for line in sequences:
                prompt = line['prompt']
                prompt_type = line['prompt_type']
                generation = line['generation']
                category = line['category']

                probabilities = [float(gen['probability']) for gen in generation]
                variance = np.var(probabilities) # calculate the variance of the probabilities
                prob = sum([gen['probability'] for gen in generation])/num_generation

                if "undoable" in dataset_name:               
                    f.write(f'{prob},0,{variance},{category}\n')
                else:    
                    f.write(f'{prob},1,{variance},{category}\n')
        else:
            f.write('probability,label,variance\n')
            for line in sequences:
                prompt = line['prompt']
                prompt_type = line['prompt_type']
                generation = line['generation']
                probabilities = [float(gen['probability']) for gen in generation]
                variance = np.var(probabilities) # calculate the variance of the probabilities
                prob = sum([gen['probability'] for gen in generation])/num_generation
                if "undoable" in dataset_name or "mix" in dataset_name:               
                    f.write(f'{prob},0,{variance}\n')
                else:    
                    f.write(f'{prob},1,{variance}\n')

#@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(15), retry_error_callback=log_retry_error)
def run_palm2(path_in, path_out,  prompt_type, dataset_name,ratio=1.0,mode="withCategory"):
    sequences = []
    num_generation = 5
    temperature = 0.7
    with open(path_in, 'r') as f:
            datas = json.load(f)

    if ratio < 1.0:
        datas = np.random.choice(datas, int(len(datas)*ratio), replace=False)
        print(len(datas))
    palm.configure(api_key=google_api_key)
    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    #model_name = models[0].name
    print("here is google model")
    
    model_name = 'models/text-bison-001'
    for entry in tqdm(datas):
        str_tmp = """You are a helpful assistant. \n"""
        prompt = str_tmp+get_prompt(model,prompt_type,dataset_name, entry["prompt"])
        retries = 0
        max_retries = 30  # 设定最大重试次数以避免无限循环
        
        while retries < max_retries:
            try:
                completion = palm.generate_text(
                            model=model_name,
                            prompt=prompt,
                            # The number of candidates to return
                            candidate_count=num_generation,
                            # Set the temperature to 1.0 for more variety of responses.
                            temperature=temperature,
                            max_output_tokens=800,
                        )
                if len(completion.candidates) < num_generation:
                    print(f"Warning: Not enough candidates generated, retrying... (Attempt {retries+1}/{max_retries})")
                    time.sleep(2)
                    retries += 1
                    continue
                time.sleep(2)
                model_outputs = [completion.candidates[i]['output'] for i in range(num_generation)]
                prob_outputs = [extract_num(model_outputs[i]) for i in range(num_generation)]
                if 'None' in prob_outputs and retries < max_retries-1:  # 检测到None值，准备重试
                    time.sleep(2)
                    retries += 1
                    print(f"Detected None in probabilities, retrying... (Attempt {retries}/{max_retries})")
                    continue  # 跳过当前循环的剩余部分，开始下一次重试
                elif 'None' in prob_outputs and retries == max_retries-1:
                    print(f"Reached max retries, proceeding with current data...")
                    for i in prob_outputs:
                        if i == 'None':
                            i = 0.5
                    retries += 1
                    break

                # 如果没有检测到None，或者已达到最大重试次数，跳出循环
                break
            except  requests.exceptions.HTTPError as e:
                if e.response.status_code == 500:
                    print(f"HTTP 500 Server Error: retrying... (Attempt {retries+1}/{max_retries})")
                    time.sleep(5)
                    retries += 1
                else:
                    raise  # Re-raise other HTTP errors not handled
            except Exception as e:
                print(f"Unexpected error: {str(e)}, retrying... (Attempt {retries+1}/{max_retries})")
                retries += 1

        if retries == max_retries:
            print("Reached max retries, proceeding with current data...")
            # 在这里可以决定如何处理达到最大重试次数的情况，比如使用默认值
                
        
        generation = [{'response':model_outputs[i],'probability':prob_outputs[i]} for i in range(num_generation)]
        if mode == "withCategory":
            sequence_dict = {
                        'prompt': line["prompt"],
                        'prompt_type': prompt_type,
                        'category': line["category"],
                        'generation': generation,
                        }
        else:
            sequence_dict = {
                        'prompt': line["prompt"],
                        'prompt_type': prompt_type,
                        'generation': generation,
                        }
     
        sequences.append(sequence_dict)
    print("Finished generation. Saving to files...\n")
    #json_file_path = path_out[:-5]+"temperature="+str(temperature)+".json"
    with open(path_out, 'w') as f:
            json.dump(sequences, f,indent=4)
    print("Saving to csv...\n")
    with open(f'{path_out[:-5]}.csv', 'w') as f:
        if mode == "withCategory":
            f.write('probability,label,variance,category\n')
            for line in sequences:
                prompt = line['prompt']
                prompt_type = line['prompt_type']
                generation = line['generation']
                category = line['category']

                probabilities = [float(gen['probability']) for gen in generation]
                variance = np.var(probabilities) # calculate the variance of the probabilities
                prob = sum([gen['probability'] for gen in generation])/num_generation

                if "undoable" in dataset_name:               
                    f.write(f'{prob},0,{variance},{category}\n')
                else:    
                    f.write(f'{prob},1,{variance},{category}\n')
        else:
            f.write('probability,label,variance\n')
            for line in sequences:
                prompt = line['prompt']
                prompt_type = line['prompt_type']
                generation = line['generation']
                probabilities = [float(gen['probability']) for gen in generation]
                variance = np.var(probabilities) # calculate the variance of the probabilities
                prob = sum([gen['probability'] for gen in generation])/num_generation
                if "undoable" in dataset_name or "mix" in dataset_name:               
                    f.write(f'{prob},0,{variance}\n')
                else:    
                    f.write(f'{prob},1,{variance}\n')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, choices=['doable_task_2k','undoable_task_2k','doable_task_gpt3.5_ablation','undoable_task_gpt3.5_ablation'], required=True)
    parser.add_argument('--prompt', type=str, choices=['pre', 'post','mid','mix','mid_no_step','style_transfer','style_transfer_no_example'], required=True)
    parser.add_argument('--model', type=str, choices=['gpt-3.5-turbo','gpt-4','palm2', 'llama2-70b',], required=True)
    parser.add_argument('--fraction', type=float, default=1.0)

    return parser.parse_args()

# main function
if __name__ == "__main__":
    
    args = vars(parse_args())
    prompt_type = args['prompt']
    model = args['model']
    dataset_name = args['data']
    ratio = args['fraction']
    
    test_in = os.path.join(_settings.DATA_FOLDER_TASK,f'{dataset_name}.json')
    
    test_out_folder= os.path.join(_settings.GENERATION_FOLDER,f'{dataset_name}')
    os.makedirs(test_out_folder, exist_ok=True)
    test_out = os.path.join(test_out_folder,f'{dataset_name}-{model}-{prompt_type}-{ratio}.json')
        

    # run the models
    if model == 'gpt-3.5-turbo':
        print("Running gpt-3.5-turbo-0125...")
        run_GPT_chat(test_in, test_out, prompt_type, dataset_name, model="gpt-3.5-turbo-0125",ratio=ratio,mode="withCategory")
        #style_transfer(test_in, test_out, prompt_type, dataset_name, model="gpt-3.5-turbo-0125")
    elif model == 'gpt-4':
        print("Running gpt-4-turbo-preview...")
        # gpt-4-turbo-preview
        run_GPT_chat(test_in, test_out, prompt_type, dataset_name, model="gpt-4-turbo-preview",ratio=ratio,mode="withCategory")
    elif model == 'palm2':
        print("Running palm2...")
        run_palm2(test_in, test_out, prompt_type, dataset_name,ratio=ratio)
    elif model == 'llama2-70b':
        print("Running llama2-70b...")
        run_llama_replicate(test_in, test_out, prompt_type, dataset_name, model="llama2-70b",ratio=ratio)