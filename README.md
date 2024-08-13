## About us
This is the code of paper ["Defining Boundaries: A Spectrum of Task Feasibility for Large Language Models"](https://arxiv.org/abs/2408.05873). 

In this paper: 
- We systematically conceptualize infeasible tasks for LLMs, providing formal definitions and categorizations that cover a spectrum of related hallucinations. 
- We establish a new dataset for task feasibility and benchmark various LLMs under the developed dataset. 
- We propose three strategies to enhance the refusal awareness of LLMs when faced with infeasible tasks, by constructing a refusal-augmented instruction tuning dataset. 

## Install libraries
```
conda create -n uq python=3.9
source activate uq
pip install -r requirements.txt
```

## Structure
```
Infeasible-Benchmark
├─ README.md
├─ _settings.py : set the path and APIs
├─ dataset
│  ├─ create_task_data.ipynb : get the 
│  ├─ generated_data : process the data generated from src/generate_task.py
│  ├─ output : save the result of src/generate.py
│  └─ raw_data
│     └─ task : task feasibility datasets
├─ prompts
│  └─ zero_shot.py : 4 prompts
├─ requirements.txt
├─ src
│  ├─ eval.py : implementation of metrics
│  ├─ finetune_data : process the data to be finetuned
│  ├─ generate.py : get the generations of LLMs
│  ├─ generate_task.py : prompts to generate data
│  ├─ post_process.py : extract probabilities from generations
│  └─ result_eval.ipynb : calculation of metrics
└─ test.sh : commands to get generations

```

## Acknowledgement
We used the toolkit : [LMFlow](https://github.com/OptimalScale/LMFlow) to fine-tune the models. 
