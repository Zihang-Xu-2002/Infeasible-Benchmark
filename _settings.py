import getpass
import os
import sys

# __USERNAME = getpass.getuser()

openai_key = "Your openai key" 
google_api_key = "Your google key"



_BASE_DIR = f'Your path to infeasible-benchmark'

DATA_FOLDER = os.path.join(_BASE_DIR, 'dataset')
DATA_FOLDER_TASK = os.path.join(DATA_FOLDER,'raw_data','task')
GENERATION_FOLDER = os.path.join(DATA_FOLDER, 'output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)
