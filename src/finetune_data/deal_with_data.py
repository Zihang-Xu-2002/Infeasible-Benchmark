import json
import random

#   Below are the code used to generate the "random" dataset
def random_dataset(data,raw_data,undoable_answer):
    random_nums = 952
    # Ensure that random_nums is not greater than the length of data
    random_nums = min(random_nums, len(data))
    # Select random_nums unique entries from data
    entries = random.sample(data, random_nums)
    for entry in entries:
        entry['answer'] = random.choice(undoable_answer)['answer']
    for entry in data:
        if 'answer' not in entry:
            for raw_entry in raw_data:
                if raw_entry['prompt'] == entry['prompt']:
                    entry['answer'] = raw_entry['answer']
                    break
    print(len(data))
    with open('./alpaca_train_random.json', 'w') as f:
        json.dump(data, f, indent=4)




#  Below are the code used to generate the "selected" dataset
def selected_dataset(data,raw_data,undoable_answer):
    count_train_undoable = 0
    data_train_undoable = []
    data_train_doable = []
    threshold = 0.5
    for entry in data :
        prob =  sum([float(gen['probability']) for gen in entry['generation']])/len(entry['generation'])
    
        if prob < threshold:
            #randomly choose an entry from undoable_answer
            entry['answer'] = random.choice(undoable_answer)['answer']
            count_train_undoable += 1
            data_train_undoable.append(entry)
        else:
            
            for raw_entry in raw_data:
                if raw_entry['prompt'] == entry['prompt']:
                    entry['answer'] = raw_entry['answer']
                    break
            data_train_doable.append(entry) 

    data_selected = data_train_doable + data_train_undoable
    print(f'Number of undoable entries in training set : {count_train_undoable}')
    print("Number of train entries", len(data_selected))
    with open('./alpaca_train_selected_gentle.json', 'w') as f:
        json.dump(data_selected, f, indent=4)




#   Below are the code used to generate the "extra" dataset
def extra_dataset(data,raw_data,undoable_answer):
    for entry in data:
        for raw_entry in raw_data:
            if raw_entry['prompt'] == entry['prompt']:
                entry['answer'] = raw_entry['answer']
                break
    #print(len(data_train))

    with open("undoable_task_2k_checked.json") as f:
        undoable_task = json.load(f)

    #randonly sample 1030 undoable entries as the extra data
    undoable_task_extra = random.sample(undoable_task, 1030) # 1030

    for entry in undoable_task_extra:
        entry['answer'] = random.choice(undoable_answer)['answer']

    data_extra = data + undoable_task_extra
    print(len(data_extra))

    with open('./alpaca_extra_train.json', 'w') as f:
        json.dump(data_extra, f, indent=4)

def main():
    with open('alpaca_train-gpt-4-pre-1.0.json') as f:
        data_train = json.load(f) # data with generations by GPT-4

    with open('alpaca_train.json') as f:
        data_train_raw = json.load(f) # row data in alpaca

    with open('alpaca_test-gpt-4-pre-1.0.json') as f:
        data_test = json.load(f) # data with generations by GPT-4

    with open('alpaca_test.json') as f:
        data_test_raw = json.load(f)  # row data in alpaca

    with open('./undoable_answer_gentle.json') as f :
        undoable_answer = json.load(f) # use undoable_answer or undoable_answer_gentle
    
    #random_dataset(data_train,data_train_raw,undoable_answer)
    #selected_dataset(data_train,data_train_raw,undoable_answer)
    extra_dataset(data_train,data_train_raw,undoable_answer)

if __name__ == '__main__':
    main()
    
        