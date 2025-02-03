set -x

# source /root/miniconda3/etc/profile.d/conda.sh
# conda init
# conda activate tg-verl

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export WANDB_PROJECT='tg-verl'
export WANDB_API_KEY='e357e4ac1b5cace6b76e7857c2d97f6a84405006'
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true

PROJECT_NAME='tg-verl'
EXPERIMENT_NAME='qwen-math-7b-prime480k-grpo-default'
DATA_PATH='/home/ubuntu/tgriggs/data/verl-prime'
SFT_MODEL_PATH='/home/ubuntu/tgriggs/models/Sky-T1-math-correct-5K-7B'
CKPT_PATH='/home/ubuntu/tgriggs/ckpts'

port=6379
ray start --head \
    --port=$port \
    --num-gpus=8 \
    --include-dashboard=false \
    --block &

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files=["$DATA_PATH/train.parquet"] \
    data.val_files=["$DATA_PATH/validation.parquet"] \
    data.train_batch_size=256 \
    data.val_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=3076 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    +actor_rollout_ref.actor.use_kl_loss=True \
    +actor_rollout_ref.actor.kl_loss_coef=0.001 \
    +actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    +actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \