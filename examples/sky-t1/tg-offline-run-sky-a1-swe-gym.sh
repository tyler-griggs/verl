set -x

PROJECT_NAME='tg-oh'
EXPERIMENT_NAME='oh-offline-32b-2node-debugging'

# DATA_PATH='/mnt/user_storage/dataset/verl-simplerl-simplerlprompt/'
DATA_PATH='/mnt/user_storage/dataset/swegym_16only'
# CKPT_PATH='/mnt/user_storage/ckpt'
CKPT_PATH='/home/ray/default/ckpt'

# SFT_MODEL_PATH='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
# SFT_MODEL_PATH='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
# SFT_MODEL_PATH='/mnt/user_storage/models/OpenHands-7B-Agent'
SFT_MODEL_PATH='/mnt/user_storage/models/OpenHands-32B-Agent'

NNODES=2
# FG_TRANSFER=False
# NUM_TRAJ=16
# MAX_ITER=35
# MAX_AGENTS=64

# actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
# actor_rollout_ref.rollout.free_cache_engine=False \
# uv run --isolated --frozen --directory . --env-file .env -m verl.trainer.main_ppo_sky \
# algorithm.adv_params.verifier_gamma=1.0 \

# uv run --isolated --frozen --directory . --env-file .env -m verl.trainer.main_ppo \
uv run --isolated --frozen --directory . --env-file .env verl/trainer/main_ppo.py \
    data.train_files=["$DATA_PATH/train.parquet"] \
    data.val_files=["$DATA_PATH/validation.parquet"] \
    data.train_batch_size=16 \
    data.max_prompt_length=30720 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=0.5 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=100 \
    trainer.resume_mode=disable \
    algorithm.adv_estimator=grpo \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \