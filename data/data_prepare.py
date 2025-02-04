# python data_prepare.py --input PRIME-RL/Eurus-2-RL-Data --output ./data/math/ --math
import os

from datasets import load_dataset, Dataset
import argparse
from tqdm import tqdm

short_system_prompt="""
Your role as an assistant involves thoroughly exploring questions through a systematic long \
        thinking process before providing the final precise and accurate solutions. This requires \
        engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, \
        backtracing, and iteration to develop well-considered thinking process. \
        Please structure your response into two main sections: Thought and Solution. \
        In the Thought section, detail your reasoning process using the specified format: \
        <|begin_of_thought|> {thought with steps separated with '\n\n'} \
        <|end_of_thought|> \
        Each step should include detailed considerations such as analisying questions, summarizing \
        relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining \
        any errors, and revisiting previous steps. \
        In the Solution section, based on various attempts, explorations, and reflections from the Thought \
        section, systematically present the final solution that you deem correct. The solution should \
        remain a logical, accurate, concise expression style and detail necessary step needed to reach the \
        conclusion, formatted as follows: \
        <|begin_of_solution|> \
        {final formatted, precise, and clear solution} \
        <|end_of_solution|> \
        Now, try to solve the following question through the above guidelines:

"""

task2ability={'Coding':'code',
              'Math':'math'}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=str, default='PRIME-RL/Eurus-2-RL-Data')
    parser.add_argument('--output', type=str, default='dataset/prime')
    parser.add_argument('--math', action='store_true')
    parser.add_argument('--coding', action='store_true')
    args=parser.parse_args()

    input_dataset = load_dataset(args.input)
    for split in ['train', 'validation']:
        output_dataset=[]
        cur_dataset = input_dataset[split]
        # idx = -1
        for data_entry in tqdm(cur_dataset):
            # if split=="train" and data_entry["data_source"] == "numina_cn_k12":
            #     idx += 1
            #     if idx % 5 != 0:
            #         continue
            if args.math and data_entry['ability']!='math':
                continue
            if args.coding and data_entry['ability']!='code':
                continue
            prompt = [{
                "role": "system",
                "content": short_system_prompt
            },
            {
                "role": "user",
                "content": data_entry['prompt'][1]['content'].replace("Present the answer in LaTex format: \\boxed{Your answer}", "Return your final response within \\boxed{{}}")
            }]
            data_entry["prompt"] = prompt
            output_dataset.append(data_entry)
        
        # add aime
        if split=='validation':
            aime_dataset = load_dataset('AI-MO/aimo-validation-aime')['train'].to_pandas()
            aime_dataset = aime_dataset[aime_dataset['url'].str.contains("2024", na=False)]
            aime_dataset = [row.to_dict() for _, row in aime_dataset.iterrows()]
            for data_entry in tqdm(aime_dataset):
                cur_data = {
                    "data_source": "aime",
                    "prompt": [
                        {
                            "role": "system",
                            "content": short_system_prompt
                        },
                        {
                            "role": "user",
                            "content": data_entry['problem'] + "\nReturn your final response within \\boxed{{}}"
                        }],
                    "ability": 'aime',
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": data_entry['answer'],
                    },
                    "extra_info": {
                        'split': 'dummy',
                        'index': 0
                    }
                }
                output_dataset.append(cur_data)
            
            # math500_dataset = load_dataset('qq8933/MATH500')
            # for data_entry in tqdm(math500_dataset['test']):
            #     cur_data = {
            #         "data_source": "math500",
            #         "prompt": [
            #             {
            #                 "role": "system",
            #                 "content": short_system_prompt
            #             },
            #             {
            #                 "role": "user",
            #                 "content": data_entry['problem'] + "\nReturn your final response within \\boxed{{}}"
            #             }],
            #         "ability": 'math500',
            #         "reward_model": {
            #             "style": "rule",
            #             "ground_truth": data_entry['answer'],
            #         },
            #         "extra_info": {
            #             'split': 'dummy',
            #             'index': 0
            #         }
            #     }
                # output_dataset.append(cur_data)
            
        print(len(output_dataset))
        output_dataset = Dataset.from_list(output_dataset)

        output_dataset.to_parquet(os.path.join(args.output, f'{split}.parquet'))
