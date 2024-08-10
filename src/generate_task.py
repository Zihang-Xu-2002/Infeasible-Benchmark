import argparse
import _settings
import logging
import os
import sys
import openai
import anthropic
import replicate
import json
from tqdm import tqdm
import random

openai.api_key = _settings.openai_key

prompt_1 = '''
Generate 300 tasks for large language models. 

Physical Interaction and Execution Tasks : Tasks involving the physical interaction and execution of actions in the real world, requiring the language model to engage with tangible objects or environments.

Examples:
	
	Jump from an airplane.

	Build a birdhouse by hand.

	Sit down on a chair.

Try to generate 300 examples of Physical Interaction and Execution Tasks based on description and examples provided for large language models. Imperative sentence only. For example:

{generated_task},
{generated_task},
...
{generated_task}

Only output examples without any illustration or another words.
'''

prompt_2 = '''
Generate 300 tasks for large language models. 

Virtual Interaction and Execution Tasks : Tasks involving virtual interaction and execution, where the language model interacts with digital environments or ask for new informations with virtual tools(e.g. search engine) to perform actions or provide responses.

Examples:

	Check current traffic conditions, road closures, and detours on Port Wakefield Road before driving.

	Show me the latest blog posts from my WordPress website.

	Help me find Super Bowl tickets online.

Try to generate 300 examples of Virtual Interaction and Execution Tasks based on description and examples provided for large language models. Imperative sentence only. For example:

{generated_task},
{generated_task},
...
{generated_task}

Only output examples without any illustration or another words.
'''

prompt_3 = '''
Generate 300 tasks for large language models. 


Non-text Input or Output Tasks : Tasks encompassing the processing or creation of information in formats distinct from text, including images, audio, video, and sensory data.
    
Examples:

        Generate a video to teach people how to basketball.

        Count the number of people who speak in a given audio.

        Describe a given image.

Try to generate 300 examples of Non-text Input or Output Tasks based on description and examples provided for large language models. Imperative sentence only. For example:

{generated_task},
{generated_task},
...
{generated_task}

Only output examples without any illustration or another words.
'''

prompt_4 = '''
Generate 300 tasks for large language models. 

Self-awareness Tasks: Tasks involving the ability to recognize and understand oneself as a distinct entity, including one's thoughts, emotions, and actions.

Examples:

    Describe your emotion when you see catastrophic incidents.

    Describe the situation you feel happy.

    Tell me who is the best statistician in your mind.

Try to generate 300 concise examples of Self-awareness Tasks based on description and examples provided for large language models. Imperative sentence only. For example:

{generated_task},
{generated_task},
...
{generated_task}

Only output examples without any illustration or another words.
'''

# task_doable_1.json对应的prompt只说了doable是能被LLM完成的任务，没有undoable的定义
# Examples:
    
#         Translate \u201cCe qui a \u00e9t\u00e9 appris est assez\u201d into English.

#         Rewrite the following sentence in the passive voice: \"He forgot to close the window.\"

#         What is the most common word in English?

# TODO 加定义
# TODO seed task 
# 几个model选一个最好的做筛选

prompt_doable = '''
Generate 1000 tasks for large language models. 

Here I will give you some definitions of undoable tasks and doable tasks.
There are four types of undoable tasks: 
 1. Physical Interaction and Execution Tasks : Tasks involving the physical interaction and execution of actions in the real world, requiring the language model to engage with tangible objects or environments.
 2. Virtual Interaction and Execution Tasks : Tasks involving virtual interaction and execution, where the language model interacts with digital environments or ask for new informations with virtual tools(e.g. search engine) to perform actions or provide responses.
 3. Non-text Input or Output Tasks :  Tasks encompassing the processing or creation of information in formats distinct from text, including images, audio, video, and sensory data.
 4. Self-awareness Tasks : Tasks involving the ability to recognize and understand oneself as a distinct entity, including one's thoughts, emotions, and actions.

Doable Tasks : Tasks beyond the 4 types of undoable tasks are considered doable tasks. Doable tasks can be completed by the language model.

Examples of doable tasks:
    {}

    {}

    {}

Try to generate 1000 concise examples of Doable Tasks based on description and examples provided for large language models. For example:

{generated_task}.
{generated_task}.
...
{generated_task}

Your generation should be as diverse as possible! Only output examples without any illustration or another words.
'''

prompt_doable_1 = '''
Generate 300 tasks for large language models. 

Here I will give you the definition of doable tasks.

Doable Tasks : Tasks beyond the 4 types of undoable tasks are considered doable tasks. Doable tasks can be completed by the language model.

Examples of doable tasks:
    {}

    {}

    {}

Try to generate 300 concise examples of Doable Tasks based on description and examples provided for large language models. For example:

{{generated_task}}.
{{generated_task}}.
...
{{generated_task}}

Your generation should be as diverse as possible! Only output examples without any illustration or another words.
'''

prompt_doable_naive = '''
Generate 1000 tasks for large language models. 

Doable Tasks : Tasks that can be completed by the language model.

Examples:
    
        Translate \u201cCe qui a \u00e9t\u00e9 appris est assez\u201d into English.

        Rewrite the following sentence in the passive voice: \"He forgot to close the window.\"

        What is the most common word in English?

Try to generate 1000 concise examples of Doable Tasks based on description and examples provided for large language models. For example:

{{generated_task}}.
{{generated_task}}.
...
{{generated_task}}

Your generation should be as diverse as possible! Only output examples without any illustration or another words.
'''

#  Generate three different customer service questions and answers.

def get_prompts(prompt,seed):
    if seed != None :
        # seed是一个被load的json文件
        # 从中任意选3个样本
        selected_samples = random.sample(seed, 3)
        print("selected_samples")
    
    prompt = prompt.format(selected_samples[0]['prompt'],selected_samples[1]['prompt'],selected_samples[2]['prompt'])
    return prompt



def append_tasks_to_json_file(output_file, new_tasks):
    try:
        # 尝试读取现有的JSON文件
        with open(output_file, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或文件不是有效的JSON，就创建一个新列表
        tasks = []

    # 将新任务追加到列表中
    tasks.extend(new_tasks)

    # 将更新后的任务列表写回文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, ensure_ascii=False, indent=4)

def generate_tasks_with_gpt_and_save(prompt, output_file, num_generation=1):
    tasks = []
    print("Here")
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        n=num_generation,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=1.0,
        max_tokens=1024,
    )
    print("111")
    generated_task = response['choices'][0]['message']['content'].split('\n')
    # for i in range(num_generation):
    #     generated_task = response['choices'][i]['message']['content']
    for element in generated_task:
    
        tasks.append({'prompt': element.strip()})

    # 将生成的任务列表保存为JSON文件
    # with open(output_file, 'w') as f:
    #     json.dump(tasks, f, ensure_ascii=False, indent=4)
    print(len(tasks))
    tasks.pop()
    print(len(tasks))
    append_tasks_to_json_file(output_file, tasks)

    print(f"Tasks generated and saved to {output_file}")

if __name__ == '__main__':
    print("Generating tasks for large language models...")
    with open('./src/alpace_remove_unanswerrable_1k.json', 'r') as f:
        seed = json.load(f)
    print(len(seed))
    prompt = get_prompts(prompt_doable,seed)
    print(prompt)
    generate_tasks_with_gpt_and_save(prompt, 'doable_raw_gpt3.5.json')