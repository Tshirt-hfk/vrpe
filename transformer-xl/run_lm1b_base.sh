#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python3 train.py \
        --cuda \
        --data ./data/one-billion-words/ \
        --dataset lm1b \
        --n_layer 18 \
        --d_model 1024 \
        --n_head 8 \
        --d_head 128 \
        --d_inner 4096 \
        --dropout 0.0 \
        --optim adam \
        --warmup_step 20000 \
        --max_step 500000 \
        --lr 0.00025 \
        --seq_len 32 \
        --attn_span 32 \
        --eval_seq_len 32 \
        --batch_size 224 \
        --multi_gpu \
        --gpu0_bsz 32 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python3 eval.py \
        --cuda \
        --data ./data/one-billion-words/ \
        --dataset lm1b \
        --batch_size 64 \
        --seq_len 32 \
        --attn_span 32 \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
