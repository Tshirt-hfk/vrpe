#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python3 train.py \
        --cuda \
        --data ./data/enwik8/ \
        --dataset enwik8 \
        --n_layer 12 \
        --d_model 512 \
        --n_head 8 \
        --d_head 64 \
        --d_inner 1024 \
        --dropout 0.1 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 400000 \
        --seq_len 512 \
        --attn_span 512 \
        --eval_seq_len 128 \
        --batch_size 22 \
        --multi_gpu \
        --gpu0_bsz 4 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python3 eval.py \
        --cuda \
        --data ./data/enwik8/ \
        --dataset enwik8 \
        --seq_len 80 \
        --attn_span 512 \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
