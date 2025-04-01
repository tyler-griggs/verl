# python data_prepare.py --input PRIME-RL/Eurus-2-RL-Data --output ./data/math/ --math
import os

from datasets import load_dataset, Dataset
import argparse
from tqdm import tqdm


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=str, default='SWE-Gym/SWE-Gym')
    parser.add_argument('--output', type=str, default='dataset/swe-gym')
    args=parser.parse_args()

    input_dataset = load_dataset(args.input)
    for split in ['train']:
        output_dataset=[]
        cur_dataset = input_dataset[split]
        idx = -1
        for data_entry in tqdm(cur_dataset):
            cur_data = {
                "prompt": data_entry['problem_statement'],
                "data_source": "swe-gym",
                "ability": 'coding',
                "instance": data_entry,
            }
            output_dataset.append(cur_data)
        
        print(len(output_dataset))
        output_dataset = Dataset.from_list(output_dataset)

        output_dataset.to_parquet(os.path.join(args.output, f'{split}.parquet'))
    
    input_dataset = load_dataset("princeton-nlp/SWE-bench_Verified")
    for split in ['validation']:
        output_dataset=[]
        cur_dataset = input_dataset["test"]
        idx = -1
        for data_entry in tqdm(cur_dataset):
            cur_data = {
                "prompt": data_entry['problem_statement'],
                "data_source": "swe-gym",
                "ability": 'coding',
                "instance": data_entry,
            }
            output_dataset.append(cur_data)
        
        print(len(output_dataset))
        output_dataset = Dataset.from_list(output_dataset)

        output_dataset.to_parquet(os.path.join(args.output, f'{split}.parquet'))
