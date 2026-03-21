export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT=cllm2_training

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

model_path="model/OneRec-1.7B"
#trajectory_file="data/data/CLLM2_openthought/merged/merged_data_v2_8_27_opencodeinstruct.jsonl"
# trajectory_file="data/data/cllm2_packed_data/merged-traj-data-oct-16-n32w16.jsonl"
deepspeed_config_file="train_deepspeed.json"
trajectory_file="outputs/fixed_windowed/*.parquet"
output_path="outputs/OneRec-1.7B-n64w8"
n_token_seq_size=64
qlora=False

torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=101 \
    --rdzv_endpoint='localhost:5666' \
    --master_port 10000 \
    03_train.py \
    --target_model_path ${model_path} \
    --data_path "${trajectory_file}" \
    --output_dir ${output_path} \
    --max_new_tokens ${n_token_seq_size} \
    --bf16 True \
    --report_to wandb \
    --do_train \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 8 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 16384 \
    --qlora ${qlora}
