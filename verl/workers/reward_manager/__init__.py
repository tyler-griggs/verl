# Copyright 2024 PRIME team and/or its affiliates
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

from .naive import NaiveRewardManager
from .prime import PrimeRewardManager

import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


# From PRIME team
async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=300.):
    loop = asyncio.get_running_loop()
    print(f"(DL Debug in reward manager init) Inside single")
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
        # print(f"(DL Debug in reward manager init) Submitting to async")
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