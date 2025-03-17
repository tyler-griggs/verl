import json
import asyncio
import uuid
from collections import deque
from typing import Any, List, Dict, Optional, Set, Callable, Tuple
import os
import pandas as pd

import torch
from tensordict import TensorDict
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
import torch.nn.functional as F

import openhands
import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from openhands.controller.agent import Agent
from openhands.controller.state.state import State, AgentState
from openhands.core.config import LLMConfig, AgentConfig, SandboxConfig, AppConfig
from openhands.core.main import create_runtime, run_controller
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.core.message_utils import (
    events_to_messages,
)
from openhands.events.action import (
    Action,
    AgentFinishAction,
    MessageAction,
)
from openhands.events.event import EventSource
from openhands.memory.condenser import Condenser
from openhands.core.config.condenser_config import (
    NoOpCondenserConfig,
)
from openhands.llm.fn_call_converter import (
    convert_fncall_messages_to_non_fncall_messages,
)
from openhands.llm.llm import LLM
from openhands.utils.prompt import PromptManager
from openhands.utils.async_utils import call_sync_from_async

DOCKER_IMAGE_PREFIX = os.environ.get('EVAL_DOCKER_IMAGE_PREFIX', 'docker.io/xingyaoww/')
logger.info(f'Using docker image prefix: {DOCKER_IMAGE_PREFIX}')

# this is for the tokenizer.apply_chat_template to be able to generate assistant masks directly
# todo: this is a hack, we should find a better way to do this
chat_template = (
        "{% for message in messages %}"
        "{% if (message['role'] != 'assistant') %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% elif (message['role'] == 'assistant')%}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% generation %}"
        "{{message['content'] + '<|im_end|>'}}"
        "{% endgeneration %}"
        "{{'\n'}}"
        "{% endif %}"
        "{% endfor %}"
    )

def convert_right_padding_to_left(input_ids, attention_mask):
    """
    Converts right-padded tensors to left-padded tensors.
    
    Args:
        input_ids (torch.Tensor): Right-padded input IDs tensor of shape [batch_size, seq_length]
        attention_mask (torch.Tensor): Right-padded attention mask tensor of shape [batch_size, seq_length]
        
    Returns:
        tuple: (left_padded_input_ids, left_padded_attention_mask)
    """
    batch_size, seq_length = input_ids.size()
    left_padded_input_ids = torch.zeros_like(input_ids)
    left_padded_attention_mask = torch.zeros_like(attention_mask)
    
    for i in range(batch_size):
        # Get the non-padded length of this sequence
        seq_len = attention_mask[i].sum().item()
        
        # Calculate the offset for left padding
        offset = seq_length - seq_len
        
        # Copy the non-padded tokens to the end
        left_padded_input_ids[i, offset:] = input_ids[i, :seq_len]
        left_padded_attention_mask[i, offset:] = 1  # Set attention mask for non-padding tokens
    
    return left_padded_input_ids, left_padded_attention_mask

def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, 'constant', pad_token_id)

# BatchManager to handle batched generation requests
class BatchManager:
    """
    Manages batched generation requests for multiple agents.
    Uses notifications from task queue to identify final batch processing.
    Uses (instance_id, trajectory_id) tuple as the agent identifier.
    """
    
    def __init__(self, vllm_engine, tokenizer, max_batch_size=4):
        self.vllm_engine = vllm_engine
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        
        # Queue for new generation requests
        self.request_queue = asyncio.Queue()

        # Event to trigger processing when queue has items
        self.processing_event = asyncio.Event()
        
        # Flag to indicate we're processing the final batch
        self.is_final_batch = False
        self.final_batch_lock = asyncio.Lock()
        
        # Don't start the processing task in __init__ - will be started in start() method
        self.processing_task = None
        self.is_running = False
    
    def start(self):
        """Start the batch manager processing loop if not already running"""
        if not self.is_running:
            try:
                # Check if we're in an event loop
                asyncio.get_running_loop()
                # Start the processing task
                self.processing_task = asyncio.create_task(self._process_requests_loop())
                self.is_running = True
                logger.info("BatchManager processing loop started")
                return True
            except RuntimeError:
                # No event loop running
                logger.error("Cannot start BatchManager: no running event loop")
                return False
    
    async def ensure_started(self):
        """Ensure the batch manager is started (safe to call from async context)"""
        if not self.is_running:
            self.start()
    
    async def add_request(self, instance_id, trajectory_id, messages):
        """Add a generation request to the batch queue"""
        # Ensure the processing loop is running
        await self.ensure_started()
        
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
        
        # Create a tuple identifier for this agent
        agent_key = (instance_id, trajectory_id)
        
        # Create a Future object for this request
        future = asyncio.Future()
        
        request = {
            'instance_id': instance_id,
            'trajectory_id': trajectory_id,
            'agent_key': agent_key,
            'input_ids': input_ids,
            'future': future,
        }
        
        # Add to the queue
        await self.request_queue.put(request)

        # Signal that we have a request to process
        self.processing_event.set()
        
        logger.info(f"Added request for agent ({instance_id}, {trajectory_id}) to queue, current queue size: {self.request_queue.qsize()}")

        return future
        
    async def get_result(self, instance_id, trajectory_id, future):
        """Wait for and retrieve the result for a specific agent"""
        # Ensure the processing loop is running
        await self.ensure_started()
        
        # Create a tuple identifier for this agent
        logger.info(f"Waiting for result for agent ({instance_id}, {trajectory_id})")
        
        import time
        while True:
            # do not yield control here, it will die
            # await asyncio.sleep(0.1)  # Simulate waiting for result
            time.sleep(0.3)
            # logger.info(f"Still waiting: Waiting for result for agent ({instance_id}, {trajectory_id})")
            if future.done():
                response = future.result()
                logger.info(f"Received result for agent ({instance_id}, {trajectory_id}): {response}")
                break
        
        return response
        
    async def notify_final_batch(self):
        """Notify the batch manager that we're processing the final batch"""
        async with self.final_batch_lock:
            self.is_final_batch = True
            logger.info("BatchManager notified that final batch is being processed")
            # Immediately trigger processing to handle any pending requests
            self.processing_event.set()
    
    async def _process_requests_loop(self):
        """Background task to continuously process batches of requests"""
        while self.is_running:
            try:
                # Wait for the processing event or timeout
                # logger.info("Waiting for requests...")
                try:
                    await asyncio.wait_for(self.processing_event.wait(), timeout=0.2)
                except asyncio.TimeoutError:
                    # No event, check if we should process anyway
                    if self.request_queue.empty():
                        continue
                
                # Clear the event
                self.processing_event.clear()
                
                # Check if we're processing the final batch
                is_final = False
                async with self.final_batch_lock:
                    is_final = self.is_final_batch
                
                # Determine how to collect requests based on whether this is the final batch
                batch_requests = []
                
                if is_final:
                    # For the final batch, process all remaining requests regardless of batch size
                    # But still respect max_batch_size by processing in chunks if needed
                    queue_size = self.request_queue.qsize()
                    batch_size = min(self.max_batch_size, queue_size)
                    
                    logger.info(f"Processing final batch: taking {batch_size} requests from queue (total: {queue_size})")
                    
                    for _ in range(batch_size):
                        request = await self.request_queue.get()
                        batch_requests.append(request)
                    
                    # If we have fewer requests than max_batch_size, pad with dummy requests
                    if batch_requests and len(batch_requests) < self.max_batch_size:
                        need_padding = self.max_batch_size - len(batch_requests)
                        logger.info(f"Padding final batch with {need_padding} dummy requests")
                        
                        # Use the first request as a template for padding
                        template_request = batch_requests[0].copy()
                        
                        # Create dummy requests to fill the batch
                        for i in range(need_padding):
                            dummy_agent_key = ('dummy', f'pad-{i}')
                            dummy_request = template_request.copy()
                            dummy_request['instance_id'] = 'dummy'
                            dummy_request['trajectory_id'] = f'pad-{i}'
                            dummy_request['agent_key'] = dummy_agent_key
                            batch_requests.append(dummy_request)
                else:
                    # For normal batches, only process if we have enough to fill a batch
                    # or if we've been waiting too long (would need additional tracking)
                    queue_size = self.request_queue.qsize()
                    
                    # Only process a full batch or nothing
                    if queue_size >= self.max_batch_size:
                        batch_size = self.max_batch_size
                        logger.info(f"Processing full batch: taking {batch_size} requests from queue (total: {queue_size})")
                        
                        for _ in range(batch_size):
                            request = await self.request_queue.get()
                            batch_requests.append(request)
                
                # Process the batch if we have any requests
                if batch_requests:
                    agent_keys = [f"({req['instance_id']}, {req['trajectory_id']})" for req in batch_requests 
                                if req['instance_id'] != 'dummy']
                    logger.info(f"Processing batch of {len(agent_keys)} real requests")
                    
                    await self._process_batch(batch_requests)
                    logger.info(f"Batch processing complete for agents: {', '.join(agent_keys)}")
                    # check if all futures are done
                    for req in batch_requests:
                        if req['instance_id'] != 'dummy':
                            assert req['future'].done(), f"Future for agent ({req['instance_id']}, {req['trajectory_id']}) not done"

            
            except Exception as e:
                logger.error(f"Error in request processing loop: {str(e)}")
                # Sleep to avoid tight loop on error
                await asyncio.sleep(0.5)
    
    async def _process_batch(self, batch_requests):
        """Process a batch of requests"""
        try:
            # Determine max length first
            max_len = max(req['input_ids'].shape[1] for req in batch_requests)

            # Pad each sequence before concatenation
            padded_inputs = []
            for req in batch_requests:
                padded = pad_sequence_to_length(
                    req['input_ids'],
                    max_seq_len=max_len,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True
                )
                padded_inputs.append(padded)

            # Then concatenate the padded sequences
            batch_input_ids = torch.cat(padded_inputs, dim=0)
            
            batch = TensorDict(
                {
                    'input_ids': batch_input_ids,
                },
                batch_size=len(batch_requests)
            )
            
            prompts = DataProto(batch=batch)
            
            # Generate responses using the engine
            response = self.vllm_engine.generate_sequences(prompts)
            
            # Process results - only for real requests, not padding requests
            real_requests = [req for req in batch_requests if req['instance_id'] != 'dummy']
            
            for i, req in enumerate(real_requests):
                agent_key = req['agent_key']
                instance_id = req['instance_id']
                trajectory_id = req['trajectory_id']
                future = req['future']
                
                if i < len(response.batch['responses']):
                    response_ids = response.batch['responses'][i]
                    response_str = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    logger.info(f"Response for agent ({instance_id}, {trajectory_id}): {response_str}")
                else:
                    response_str = ""  # Handle case where response is missing
                
                # Store result by agent key
                # Inside _process_batch method where you store results
                if future:
                    assert not future.done(), f"Future for agent ({instance_id}, {trajectory_id}) already done"
                    future.set_result(response_str)

                logger.info(f"Completed response for agent ({instance_id}, {trajectory_id})")
                current_loop = asyncio.get_running_loop()
                logger.info(f"process batch in event loop: {id(current_loop)}")
            
            # Mark all processed real requests as done
            for req in real_requests:
                self.request_queue.task_done()
                
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            # Handle the error by adding empty responses for all real agents in the batch
            real_requests = [req for req in batch_requests if req['instance_id'] != 'dummy']
            for req in real_requests:
                if not req['future'].done():
                    req['future'].set_result("")
                self.request_queue.task_done()
    
    def close(self):
        """Clean up resources"""
        self.is_running = False
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()

def codeact_user_response(
    state: State,
    encapsulate_solution: bool = False,
    try_parse: Callable[[Action], str] | None = None,
) -> str:
    encaps_str = (
        (
            'Please encapsulate your final answer (answer ONLY) within <solution> and </solution>.\n'
            'For example: The answer to the question is <solution> 42 </solution>.\n'
        )
        if encapsulate_solution
        else ''
    )
    msg = (
        'Please continue working on the task on whatever approach you think is suitable.\n'
        'If you think you have solved the task, please first send your answer to user through message and then finish the interaction.\n'
        f'{encaps_str}'
        'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
    )

    if state.history:
        # check if the last action has an answer, if so, early exit
        if try_parse is not None:
            last_action = next(
                (
                    event
                    for event in reversed(state.history)
                    if isinstance(event, Action)
                ),
                None,
            )
            ans = try_parse(last_action)
            if ans is not None:
                return '/exit'

        # check if the agent has tried to talk to the user 3 times, if so, let the agent know it can give up
        user_msgs = [
            event
            for event in state.history
            if isinstance(event, MessageAction) and event.source == 'user'
        ]
        if len(user_msgs) >= 2:
            # let the agent know that it can give up when it has tried 3 times
            return (
                msg
                + 'If you want to give up, use the "finish" tool to finish the interaction.\n'
            )
    return msg

def get_instance_docker_image(instance_id: str) -> str:
    image_name = 'sweb.eval.x86_64.' + instance_id
    image_name = image_name.replace(
        '__', '_s_'
    )  # to comply with docker image naming convention
    return (DOCKER_IMAGE_PREFIX.rstrip('/') + '/' + image_name).lower()

# Helper function for sandbox config
def get_default_sandbox_config_for_eval():
    return SandboxConfig(
        use_host_network=False,
        timeout=300,
        api_key=os.environ.get('ALLHANDS_API_KEY', None),
        remote_runtime_api_url=os.environ.get('SANDBOX_REMOTE_RUNTIME_API_URL'),
        keep_runtime_alive=False,
        remote_runtime_init_timeout=3600,
        remote_runtime_api_timeout=120,
        remote_runtime_enable_retries=True,
        remote_runtime_class='sysbox',
    )

class OnlineCodeActAgent(Agent):
    """
    An online implementation of CodeActAgent that leverages vLLM's asynchronous capabilities
    for a single agent instance.
    """
    
    def __init__(
        self,
        instance_id: int,
        trajectory_id: int,
        batch_manager: BatchManager,
        max_prompt_length: int = 1024,
        tokenizer=None,
    ) -> None:
        """
        Initialize a single OnlineCodeActAgent instance.
        """
        # dummy value to let openhands tracks the name
        llm = LLM(LLMConfig(model="dummy"))

        super().__init__(llm, AgentConfig())
        
        self.batch_manager = batch_manager
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.reset()
        self.step_count = 0
        
        # Store instance and trajectory IDs separately
        self.instance_id = instance_id
        self.trajectory_id = trajectory_id
        
        # Initialize tools
        self.tools = codeact_function_calling.get_tools(
            codeact_enable_browsing=False,
            codeact_enable_jupyter=False,
            codeact_enable_llm_editor=False,
        )
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager(
            microagent_dir=os.path.join(
                os.path.dirname(os.path.dirname(openhands.__file__)),
                'microagents',
            ),
            prompt_dir=os.path.join(os.path.dirname(openhands.agenthub.codeact_agent.__file__), 'prompts'),
            disabled_microagents=None,
        )
        
        # Initialize condenser
        self.condenser = Condenser.from_config(NoOpCondenserConfig())
        
        # Initialize state
        self.pending_actions = deque()
        
        # will be set in _initialize_runtime_for_agent
        self.runtime = None
        self.instruction = None
        self.config = None

    def close(self):
        """Close the agent runtime."""
        if self.runtime:
            self.runtime.close()
        
    def _initial_messages(self) -> list[Message]:
        """Creates the initial messages (including the system prompt) for the LLM conversation."""
        return [
            Message(
                role='system',
                content=[
                    TextContent(
                        text=self.prompt_manager.get_system_message(),
                        cache_prompt=False,  # Assuming caching is active
                    )
                ],
            )
        ]
        
    def _enhance_messages(self, messages: list[Message]) -> list[Message]:
        """Enhances the user message with additional context based on keywords matched."""
        results: list[Message] = []
        is_first_message_handled = False

        for msg in messages:
            if msg.role == 'user' and not is_first_message_handled:
                is_first_message_handled = True
                # Compose the first user message with examples
                self.prompt_manager.add_examples_to_initial_message(msg)

                # Add repo/runtime info if enabled
                if self.config.get_agent_config().enable_prompt_extensions:
                    self.prompt_manager.add_info_to_initial_message(msg)

            # Enhance the user message with additional context based on keywords matched
            if msg.role == 'user':
                self.prompt_manager.enhance_message(msg)

            results.append(msg)

        return results
        
    def _get_messages(self, state: State) -> List[Message]:
        """Get the message history for this agent."""
        # Start with initial messages (system prompt)
        messages = self._initial_messages()
        
        # If using a condenser, condense the history
        events = self.condenser.condensed_history(state)
        
        # Convert history events to messages
        messages += events_to_messages(
            events,
            max_message_chars=32768,  # Default value, adjust as needed
            vision_is_active=False,  # Assuming vision is not active
            enable_som_visual_browsing=False,  # Assuming SOM visual browsing is not enabled
        )
        
        messages = self._enhance_messages(messages)
        
        return messages

    # Conversion utility function
    def convert_str_to_completion_format(self, response_str):
        from types import SimpleNamespace
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(
                        content=response_str,
                        role="assistant",
                        tool_calls=None,
                        function_calling=None
                    )
                )
            ]
        )

    async def step(self, state: State) -> Action:
        """Generate a response using batched vLLM."""
        self.step_count += 1
        print(f"instance id {self.instance_id}, trajectory {self.trajectory_id}, step {self.step_count}")
        if self.pending_actions:
            return self.pending_actions.popleft()

        # if we're done, go back
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        # prepare what we want to send to the LLM
        messages = self._get_messages(state)
        messages = self.llm.format_messages_for_llm(messages)
        messages = convert_fncall_messages_to_non_fncall_messages(
                    messages, self.tools
                )
        # print(f"instance id {self.instance_id}, trajectory {self.trajectory_id}, messages: {messages}")
        
        try:
            # Add request to batch manager using instance_id and trajectory_id
            future = await self.batch_manager.add_request(
                self.instance_id,
                self.trajectory_id,
                messages
            )
            
            # Wait for the result
            response_str = await self.batch_manager.get_result(
                self.instance_id,
                self.trajectory_id,
                future
            )
            
            # logger.info(f"instance id {self.instance_id}, trajectory {self.trajectory_id}, Response: {response_str}")
            
            if not response_str:
                # If we got an empty response (possible error), return a message action
                self.pending_actions.append(
                    MessageAction(
                        content="I encountered an error processing your request. Let's try again.",
                        source=EventSource.AGENT
                    )
                )
            else:
                # Convert to actions
                actions = codeact_function_calling.response_to_actions(
                    self.convert_str_to_completion_format(response_str)
                )
                logger.info(f"Take actions: {actions}")
                
                for action in actions:
                    self.pending_actions.append(action)
        
        except Exception as e:
            logger.error(f"Error in agent step: {str(e)}")
            # Handle errors gracefully by creating a message action
            self.pending_actions.append(
                MessageAction(
                    content=f"I encountered an error: {str(e)}. Let's try a different approach.",
                    source=EventSource.AGENT
                )
            )
        
        # Return the first pending action
        if not self.pending_actions:
            # Fallback in case of empty actions
            return MessageAction(
                content="I'm unable to proceed. Please provide more information.",
                source=EventSource.AGENT
            )
            
        return self.pending_actions.popleft()
    
    def get_final_messages(self, state: State) -> List[Message]:
        """Get the final messages for this agent."""
        messages = self._get_messages(state)
        messages = self.llm.format_messages_for_llm(messages)
        messages = convert_fncall_messages_to_non_fncall_messages(
                    messages, self.tools
                )
        return messages
    

Agent.register('OnlineCodeActAgent', OnlineCodeActAgent) 

class CodeActAgentGroup:
    """
    A class that manages multiple CodeActAgent instances to generate trajectories in parallel.
    """
    
    def __init__(
        self,
        batch: DataProto,
        num_trajectories: int,
        vllm_engine: Any,
        max_prompt_length: int = 1024,
        max_parallel_agents: int = 1,
        max_iterations: int = 10,
        tokenizer: Any = None,
    ) -> None:
        """
        Initialize the CodeActAgentGroup to manage multiple agent instances.
        
        Args:
            batch: DataProto containing the batch of data
            num_trajectories: Number of trajectories to generate per instance
            vllm_engine: The vLLM engine for generation
            max_prompt_length: Maximum prompt length
            max_parallel_agents: Maximum number of agents to run in parallel
            max_iterations: Maximum number of iterations per agent
            tokenizer: Tokenizer to use for text encoding/decoding
            max_batch_size: Maximum batch size for LLM generation
        """
        self.batch = batch
        self.vllm_engine = vllm_engine
        self.max_prompt_length = max_prompt_length
        self.max_parallel_agents = max_parallel_agents
        self.max_iterations = max_iterations
        self.num_trajectories = num_trajectories
        self.tokenizer = tokenizer
        
        # Create a batch manager for coordinating batched generations
        # Use max_batch_size parameter instead of fixed batch_size
        self.batch_manager = BatchManager(
            vllm_engine=vllm_engine,
            tokenizer=tokenizer,
            max_batch_size=self.max_parallel_agents
        )
        
        # Map of instance ID to agent instance
        self.agents = {}
        
        # Map of instance ID to agent results
        self.results = {}
        
        # Initialize agents for each instance
        self._initialize_agents()
    

    def _convert_results_to_dataproto(self) -> DataProto:
        """
        Convert results to DataProto format for training.
        
        Args:
            results: Dictionary of results, with structure {instance_id: {trajectory_id: result_dict}}
            input_dataproto: The input DataProto that contains the original batch data
            tokenizer: The tokenizer to use for encoding messages
            
        Returns:
            DataProto: A DataProto object with the converted results
        """

        # Non-tensor data
        git_patch_list = []
        success_list = []
        error_list = []
        
        # Create a mapping of instance_id -> list of trajectories
        instance_trajectories = {}
        for instance_id, trajectories in self.results.items():
            instance_trajectories[instance_id] = []
            for trajectory_id, result in trajectories.items():
                instance_trajectories[instance_id].append(result)

        # Create the final results in the same order as the batch
        matched_results = []
        instance_list = []
        for batch_item in self.batch:
            instance_id = batch_item.non_tensor_batch['instance']['instance_id']
            instance = batch_item.non_tensor_batch['instance']
            if instance_id in instance_trajectories:
                # Add all trajectories for this instance
                traj_results = instance_trajectories[instance_id]
                matched_results.extend(traj_results)
                instance_list.extend([instance] * len(traj_results))
        
        assert len(matched_results) == self.num_trajectories * len(self.batch), f"Expected number of results {self.num_trajectories * len(self.batch)}, got {len(matched_results)}"
        # Get batch of messages
        all_messages = []
        all_prompts = []
        all_responses = []
        for result in matched_results:
            messages = result.get('messages', [])
            all_messages.append(messages)
            # get the response: starting from the first assistant message
            starting_index = 0
            for i, msg in enumerate(messages):
                if msg["role"] == 'assistant':
                    starting_index = i
                    break
            prompt = messages[:starting_index]
            all_prompts.append(prompt)
            response = messages[starting_index:]
            all_responses.append(response)


            # Also add non-tensor data
            git_patch_list.append(result.get('git_patch', None))
            success_list.append(not result.get('success', True))  # Inverting as per original requirement
            error_list.append(result.get('error', None))
        
        # Encode messages, get assitant mask and position ids
        prompt_encodings = self.tokenizer.apply_chat_template(
            all_prompts, 
            return_tensors="pt",
            add_generation_prompt=False,
            return_dict=True,
            padding=True
        )
        prompt_input_ids = prompt_encodings['input_ids']
        prompt_attention_mask = prompt_encodings['attention_mask']
        prompt_input_ids, prompt_attention_mask = convert_right_padding_to_left(prompt_input_ids, prompt_attention_mask)

        response_encodings = self.tokenizer.apply_chat_template(
            all_responses,
            chat_template=chat_template,
            return_tensors="pt",
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
            return_dict=True,
            padding=True
        )
        response_ids = response_encodings['input_ids']
        response_attention_mask = response_encodings['attention_mask']
        response_assistant_mask = torch.tensor(response_encodings['assistant_masks'])
        
        input_ids = torch.cat([prompt_input_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)

        # Create tensor dictionary
        print(f"input_ids shape: {input_ids.shape}, response_ids shape: {response_ids.shape}")
        assert input_ids.shape[1] == attention_mask.shape[1] == position_ids.shape[1], f"input_ids shape {input_ids.shape}, attention_mask shape {attention_mask.shape}, position_ids shape {position_ids.shape} do not match"
        assert response_ids.shape[1] == response_assistant_mask.shape[1], f"response_ids shape {response_ids.shape}, response_assistant_mask shape {response_assistant_mask.shape} do not match"
        tensor_dict = {
            'input_ids': input_ids,
            'responses': response_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': response_assistant_mask,
        }
        
        # Create non-tensor dictionary
        non_tensor_dict = {
            'git_patch': git_patch_list,
            'success': success_list,
            'error': error_list,
            'instance': instance_list
        }
        
        # Create and return DataProto
        result_dataproto = DataProto.from_dict(
            tensors=tensor_dict,
            non_tensors=non_tensor_dict
        )
        
        return result_dataproto
        
    async def initialize_batch_manager(self):
        """Initialize the batch manager when inside an async context"""
        if hasattr(self, 'batch_manager'):
            # Start the batch manager's processing loop
            await self.batch_manager.ensure_started()
        
    def close(self):
        """Clean up resources"""
        # Close the batch manager
        if hasattr(self, 'batch_manager'):
            self.batch_manager.close()
            
        # Close all agent instances
        for instance_id in self.agents:
            for trajectory_id in self.agents[instance_id]:
                try:
                    self.agents[instance_id][trajectory_id].close()
                except Exception as e:
                    logger.warning(f"Error closing agent {instance_id}, trajectory {trajectory_id}: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up"""
        self.close()
    
    def _initialize_agents(self) -> None:
        """Initialize agent instances for each task."""
        for data_item in self.batch:
            instance_id = data_item.non_tensor_batch['instance']['instance_id']
            self.agents[instance_id] = {}
            for n in range(self.num_trajectories):
                self.agents[instance_id][n] = OnlineCodeActAgent(
                    instance_id=instance_id,
                    trajectory_id=n,
                    batch_manager=self.batch_manager,  # Using batch manager instead of direct vLLM access
                    max_prompt_length=self.max_prompt_length,
                    tokenizer=self.tokenizer,
                )
                # Set the instance data for each agent
                self.agents[instance_id][n].instance_data = data_item.non_tensor_batch['instance']
                self.agents[instance_id][n].max_iterations = self.max_iterations
    
    async def _initialize_runtime_for_agent(self, batch_id: int, trajectory_id: int) -> None:
        """Initialize the runtime for a specific agent."""
        instance_id = self.batch[batch_id].non_tensor_batch['instance']['instance_id']
        instance = pd.Series(self.batch[batch_id].non_tensor_batch['instance'])
        agent = self.agents[instance_id][trajectory_id]
        
        try:
            # Configure sandbox
            RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'
            SWE_BENCH_CONTAINER_IMAGE = 'ghcr.io/opendevin/eval-swe-bench:full-v1.2.1'
            
            if os.environ.get('USE_INSTANCE_IMAGE', 'true').lower() == 'true':
                # Use a different instance image for each instance of swe-bench eval
                base_container_image = get_instance_docker_image(instance_id)
                logger.info(
                    f'Using instance container image: {base_container_image}. '
                    f'Please make sure this image exists. '
                    f'Submit an issue on https://github.com/All-Hands-AI/OpenHands if you run into any issues.'
                )
            else:
                base_container_image = SWE_BENCH_CONTAINER_IMAGE
                logger.info(f'Using swe-bench container image: {base_container_image}')
            
            sandbox_config = get_default_sandbox_config_for_eval()
            sandbox_config.base_container_image = base_container_image
            sandbox_config.enable_auto_lint = True
            sandbox_config.use_host_network = False
            sandbox_config.platform = 'linux/amd64'
            
            app_config = AppConfig(
                default_agent='OnlineCodeActAgent',
                run_as_openhands=False,
                max_iterations=self.max_iterations,
                runtime='remote',
                sandbox=sandbox_config,
                workspace_base=None,
                workspace_mount_path=None,
            )
            agent_config = AgentConfig(
                codeact_enable_jupyter=False,
                codeact_enable_browsing=False,
                codeact_enable_llm_editor=False,
                condenser=NoOpCondenserConfig(),
                enable_prompt_extensions=False,
            )
            app_config.set_agent_config(agent_config)
            agent.config = app_config
            
            # Create runtime
            runtime = create_runtime(app_config)
            
            # Connect runtime
            await runtime.connect()
            
            # Initialize runtime
            from .utils import initialize_runtime, get_instruction
            # initialize_runtime(runtime, instance)
            await call_sync_from_async(initialize_runtime, runtime, instance)
            
            # Store the runtime and instruction
            agent.runtime = runtime
            agent.instruction = get_instruction(instance)
            
            logger.info(f"Successfully initialized runtime for instance {instance_id}")
        except Exception as e:
            logger.error(f"Failed to initialize runtime for instance {instance_id}: {str(e)}")
            if 'runtime' in locals() and runtime:
                runtime.close()
            
            # Update agent state to reflect error
            agent.error = str(e)
            agent.agent_state = AgentState.ERROR
            raise
    
    async def _run_agent(self, batch_id: int, trajectory_id: int) -> Dict[str, Any]:
        instance_id = self.batch[batch_id].non_tensor_batch['instance']['instance_id']
        """Run a single agent to completion and return the results."""
        agent = self.agents[instance_id][trajectory_id]
        assert agent is not None
        instance = pd.Series(self.batch[batch_id].non_tensor_batch['instance'])
        runtime = agent.runtime
        
        try:
            # Run the agent controller
            state = await run_controller(
                config=agent.config,
                initial_user_action=MessageAction(content=agent.instruction),
                runtime=runtime,
                agent=agent,
                fake_user_response_fn=codeact_user_response,
            )

            if state:
                print(state.last_error)
            
            from .utils import complete_runtime, is_fatal_evaluation_error
            # Check for fatal errors
            if state and is_fatal_evaluation_error(state.last_error):
                logger.error(f"Fatal error in agent {instance_id}: {state.last_error}")
                raise Exception('Fatal error detected: ' + state.last_error)
            
            final_messages = agent.get_final_messages(state)
            # Complete the runtime and get the git patch
            return_val = await call_sync_from_async(complete_runtime, runtime, instance)
            # return_val = complete_runtime(runtime, instance)
                
            return {
                'instance_id': instance_id,
                'trajectory_id': trajectory_id,
                'state': state,
                'git_patch': return_val.get('git_patch', None),
                'messages': final_messages,
                'success': not bool(state.last_error if state else True),
                'error': state.last_error if state and state.last_error else None
            }
        except Exception as e:
            logger.error(f"Error running agent {instance_id}: {str(e)}")
            # Update agent state to reflect error
            agent.error = str(e)
            agent.agent_state = AgentState.ERROR
            
            return {
                'instance_id': instance_id,
                'trajectory_id': trajectory_id,
                'messages': [],
                'state': None,
                'git_patch': None,
                'success': False,
                'error': str(e)
            }
        finally:
            # Clean up the runtime
            if runtime:
                try:
                    runtime.close()
                except Exception as e:
                    logger.warning(f"Error closing runtime for instance {instance_id}: {str(e)}")
    
    async def generate_trajectories(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """
        Generate trajectories for all instances using async with continuous batching.
        
        This implementation maintains a constant number of active tasks by launching
        new ones immediately as previous ones complete, while leveraging the batched
        LLM generation through the shared BatchManager.
        
        Returns:
            Dict mapping instance ID to a dict of trajectory ID to results
        """
        total_instances = len(self.batch)
        print("Total instances:", total_instances)
        
        # Initialize the batch manager's processing task
        await self.initialize_batch_manager()
        
        # Create a queue of all (instance_id, trajectory_id) pairs
        task_queue = asyncio.Queue()
        for batch_idx in range(total_instances):
            for trajectory_id in range(self.num_trajectories):
                await task_queue.put((batch_idx, trajectory_id))
        
        # Keep track of active tasks
        active_tasks = set()
        
        # Helper function to process one task
        async def process_one_task():
            current_loop = asyncio.get_running_loop()
            logger.info(f"process_one_task started in event loop: {id(current_loop)}")
            batch_idx, trajectory_id = await task_queue.get()
            instance_id = self.batch[batch_idx].non_tensor_batch['instance']['instance_id']
            try:    
                # Initialize runtime and run agent
                logger.info(f"Initializing runtime for instance {instance_id}, trajectory {trajectory_id}")
                await self._initialize_runtime_for_agent(batch_idx, trajectory_id)
                
                logger.info(f"Running agent for instance {instance_id}, trajectory {trajectory_id}")
                result = await self._run_agent(batch_idx, trajectory_id)
                
                # Store the result
                if instance_id not in self.results:
                    self.results[instance_id] = {}
                self.results[instance_id][trajectory_id] = result
                
                logger.info(f"Successfully completed instance {instance_id}, trajectory {trajectory_id}")
                
            except Exception as e:
                logger.error(f"Error processing instance {instance_id}, trajectory {trajectory_id}: {str(e)}")
                # Store error result
                if instance_id not in self.results:
                    self.results[instance_id] = {}
                self.results[instance_id][trajectory_id] = {
                    'instance_id': instance_id,
                    'trajectory_id': trajectory_id,
                    'state': None,
                    'git_patch': None,
                    'success': False,
                    'error': str(e)
                }
            finally:
                # Clean up any resources
                try:
                    agent = self.agents.get(instance_id, {}).get(trajectory_id)
                    if agent:
                        agent.close()
                except Exception as cleanup_error:
                    logger.warning(f"Error during cleanup for {instance_id}, trajectory {trajectory_id}: {cleanup_error}")
                
                # Mark task as done
                logger.info(f"Task done for instance {instance_id}, trajectory {trajectory_id}")
                task_queue.task_done()
                
                # Important: Launch a new task to replace this one
                if not task_queue.empty():
                    task = asyncio.create_task(process_one_task())
                    active_tasks.add(task)
                    task.add_done_callback(lambda t: active_tasks.discard(t))
                else:
                    # Queue is empty - this indicates we're processing the final batch
                    logger.info("Task queue is empty - processing the final batch of tasks")
                    # Notify the batch manager that this is the final batch
                    await self.batch_manager.notify_final_batch()
        
        # Use the max_parallel_agents directly as our parallelism level
        # This ensures we always have exactly max_parallel_agents tasks running
        optimal_parallelism = self.max_parallel_agents
        
        logger.info(f"Starting with {optimal_parallelism} parallel tasks")
        
        # Start initial batch of tasks, up to queue size
        print(f"Initial queue size: {task_queue.qsize()}")
        for _ in range(min(optimal_parallelism, task_queue.qsize())):
            task = asyncio.create_task(process_one_task())
            active_tasks.add(task)
            task.add_done_callback(lambda t: active_tasks.discard(t))
        
        # Wait for all tasks to complete
        if task_queue.qsize() > 0:
            await task_queue.join()
        
        # Wait for any remaining active tasks
        if active_tasks:
            logger.info(f"Waiting for {len(active_tasks)} remaining tasks to complete")
            await asyncio.wait(active_tasks)
        
        results_dataproto = self._convert_results_to_dataproto()
        return results_dataproto
    
    def run(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """
        Run the agent group synchronously by creating a new event loop if necessary.
        
        Returns:
            Dict mapping instance ID to a dict of trajectory ID to results
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop exists in this thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the generate_trajectories coroutine in the event loop
        try:
            return loop.run_until_complete(self.generate_trajectories())
        finally:
            # Close the batch manager to ensure cleanup
            self.close()
            # loop.close()