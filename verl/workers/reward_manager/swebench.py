# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import statistics
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch

import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# From PRIME team
async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(evaluation_func, task, completion, reference, task_extra_info)  # Ensure synchronous
                ),
                timeout=timeout)
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"Error processing completion: {completion[:100]}, Error: {e}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(evaluation_func,
                                       completions,
                                       references,
                                       tasks,
                                       extra_info=None,
                                       num_processes=64):
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        if extra_info is None:
            extra_info = [None] * len(tasks)
        # Create tasks for all rows
        """
        results = []
        for completion, reference, task, task_extra_info in zip(completions, references, tasks, extra_info):
            print(f"Using eval func: {evaluation_func}")
            cur_score = evaluation_func(task, completion, reference, task_extra_info)
            results.append([cur_score])
            print(f"Get score: {cur_score}")
        """
            # results.append(single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.)) 
        # """
        tasks_async = [
            single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.)
            for completion, reference, task, task_extra_info in zip(completions, references, tasks, extra_info)
        ]
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except:
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print('shut down failed: ' + str(kill_err))
            raise
        # """

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks):
        # print(f"Looping result: {result}")
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result[0], (int, float, bool)):
            scores.append(float(result[0]))
        else:
            scores.append(float(result[0][0]))
    return scores


class SWEBenchRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, config=None) -> None:
        self.data_source = "SWE-Gym/SWE-Gym"
        from swegym.harness.test_spec import (
            SWEbenchInstance,
            make_test_spec,
        )
        from swegym.harness.utils import load_swebench_dataset
        full_dataset = load_swebench_dataset(self.data_source, "train")
        self.instance_id_to_test_spce_map = {}
        for instance in full_dataset:
            self.instance_id_to_test_spce_map[instance["instance_id"]] = make_test_spec(instance)

        
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.config = config

    def verify(self, data):
        response_str = data.non_tensor_batch["git_patch"]
        ground_truth = [self.instance_id_to_test_spce_map[instance['instance_id']] for instance in data.non_tensor_batch["instance"]]
        extra_info = [{"instance_id": instance['instance_id']} for instance in data.non_tensor_batch["instance"]]
        data_source = [self.data_source] * len(response_str)
        try:
            # (Dacheng): Check how to set the number of processes
            score = asyncio.run(
                parallel_compute_score_async(self.verifier_func,
                                             response_str,
                                             ground_truth,
                                             data_source,
                                             extra_info,
                                             num_processes=64))
        except asyncio.TimeoutError as e:
            print(f'Global timeout in reward computing! Setting all as 0. Error: {e}')
            score = [0. for _ in range(len(response_str))]
        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            score = [0. for _ in range(len(response_str))]
        data.batch['acc'] = torch.tensor(score, dtype=torch.float32, device=data.batch['responses'].device)
        reward_metrics = {}
        for ability in list(set(data.non_tensor_batch['ability'])):
            score_ = [data.batch['acc'][i].item() for i in range(len(data.batch['acc'])) if
                      data.non_tensor_batch['ability'][i] == ability]
            reward_metrics[f'{ability}'] = statistics.mean(score_)
        reward_metrics['all'] = data.batch['acc'].mean().item()
        
        return score, reward_metrics

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        reward_tensor_dict={}
        reward_metrics={}
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        verifier_reward=torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # prompt_ids = data.batch['prompts']
        # response_ids = data.batch['responses']
        # prompt_length = prompt_ids.shape[-1]
        # valid_response_length = data.batch['attention_mask'][:, :].sum(-1)
        # response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        # if the batch already contains evaluation results, the verification is skipped here.
        if 'acc' in data.batch:
            verifier_score = data.batch['acc'].cpu().numpy().tolist()
        else:
            verifier_score, verifier_metrics = self.verify(data)
            reward_metrics.update(verifier_metrics)
        for i in range(verifier_reward.shape[0]):
            verifier_reward[i] += verifier_score[i]

        reward_tensor_dict['gt_scores'] = verifier_reward
        
        if 'rm_scores' in data.batch.keys():
            reward_tensor_dict['rm_scores'] = data.batch['rm_scores']
            reward_metrics['reward_model']=data.batch['rm_scores'].sum(dim=1).mean().item()
            if self.config.reward_model.rm_coef!=0:
                reward_tensor += self.config.reward_model.rm_coef * reward_tensor_dict['rm_scores']

        if self.config.verifier.reward_coef!=0:
            reward_metrics['verifier'] = reward_tensor_dict['gt_scores'].sum(dim=1).mean().item()
            reward_tensor += self.config.verifier.reward_coef * reward_tensor_dict['gt_scores']

        reward_tensor_dict['all'] = reward_tensor
        reward_metrics['reward_all'] = reward_tensor.sum(dim=-1).mean(dim=0).item()

        return reward_tensor_dict, reward_metrics