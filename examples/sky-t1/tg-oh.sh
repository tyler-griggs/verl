set -x
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate sky-t1-rl
# export HF_HOME="/shared/sycao/hf_cache"
# export NCCL_DEBUG=WARN
# export WANDB_API_KEY='e357e4ac1b5cace6b76e7857c2d97f6a84405006'
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TOKENIZERS_PARALLELISM=true
# export ALLHANDS_API_KEY="sycao-sandbox-remote"
# export SANDBOX_REMOTE_RUNTIME_API_URL="http://150.136.52.109:8000"
# export SANDBOX_REMOTE_RUNTIME_API_URL="http://aa5c04af510464a9aa96b1c17a6f178c-702829876.us-west-2.elb.amazonaws.com:3000/"
# export LOG_LEVEL=ERROR


PROJECT_NAME='tg-oh'
EXPERIMENT_NAME='oh-offline-32b-2node-debugging'
# EXPERIMENT_NAME='oh-offline-32b-2node-util80-traj16-maxiter35-agents64-vllmV0-063-autotransfer'
# DATA_PATH='/mnt/user_storage/dataset/pruned_swe-gym/'
DATA_PATH='/mnt/user_storage/dataset/swe-gym2/'
# CKPT_PATH='/mnt/user_storage/ckpt'
CKPT_PATH='/home/ray/default/ckpt'
# SFT_MODEL_PATH='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
# SFT_MODEL_PATH='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
# SFT_MODEL_PATH='/mnt/user_storage/models/OpenHands-7B-Agent'
SFT_MODEL_PATH='/mnt/user_storage/models/OpenHands-32B-Agent'

# TODO: remove skip first step
# TODO: move order of log prob calculation back to before adv/reward
# TODO: remove skip optimizer save and data loader

NNODES=2
FG_TRANSFER=False
NUM_TRAJ=2
MAX_ITER=2
MAX_AGENTS=32

# actor_rollout_ref.actor.optim.lr=1e-6 \
    # actor_rollout_ref.rollout.enable_chunked_prefill=False \
    # actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    # actor_rollout_ref.async_actor_rollout=False \
    # actor_rollout_ref.rollout.fine_grain_transfer=$FG_TRANSFER \

uv run --isolated --frozen --directory . --env-file .env -m verl.trainer.main_ppo_sky \
    data.train_files=["$DATA_PATH/train.parquet"] \
    data.val_files=["$DATA_PATH/validation.parquet"] \
    data.train_batch_size=16 \
    data.max_prompt_length=30720 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.masking=True \
    actor_rollout_ref.actor.optim.lr=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_trajectories=$NUM_TRAJ \
    actor_rollout_ref.rollout.temperature=0.5 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.max_parallel_agents=$MAX_AGENTS \
    actor_rollout_ref.rollout.max_iterations=$MAX_ITER \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=0 \
    trainer.test_freq=0 \
    trainer.total_epochs=10 \
    trainer.resume_mode=disable \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \