set -x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate skygym
export NCCL_DEBUG=WARN
export WANDB_API_KEY='e357e4ac1b5cace6b76e7857c2d97f6a84405006'
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export ALLHANDS_API_KEY="sycao-sandbox-remote"
export SANDBOX_REMOTE_RUNTIME_API_URL="http://150.136.52.109:8000"
# export SANDBOX_REMOTE_RUNTIME_API_URL="http://aa5c04af510464a9aa96b1c17a6f178c-702829876.us-west-2.elb.amazonaws.com:3000/"
export LOG_LEVEL=ERROR

PROJECT_NAME='agents-rl-debug'
EXPERIMENT_NAME='swe-gym-test-verl'
DATA_PATH='/sky-t1-rl/data/swe-gym'
SFT_MODEL_PATH=/models/OpenHands-7B-Agent
CKPT_PATH='/ckpt'

port=6379
ray start --head \
    --port=$port \
    --num-gpus=8 \
    --include-dashboard=false \
    --block &

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=["$DATA_PATH/train.parquet"] \
    data.val_files=["$DATA_PATH/validation.parquet"] \
    data.train_batch_size=8 \
    data.max_prompt_length=30000 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.masking=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_trajectories=2 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.max_parallel_agents=16 \
    actor_rollout_ref.rollout.max_iterations=10 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager="swebench" \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=0 \
    trainer.total_epochs=2 \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAM $@