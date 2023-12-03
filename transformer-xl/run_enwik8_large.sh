#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ./data/enwik8/ \
        --dataset enwik8 \
        --n_layer 24 \
        --d_model 1024 \
        --n_head 8 \
        --d_head 128 \
        --d_inner 3072 \
        --dropout 0.15 \
        --dropatt 0.15 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 4000 \
        --max_step 400000 \
        --seq_len 768 \
        --attn_span 768 \
        --eval_seq_len 128 \
        --batch_size 64 \
        --multi_gpu \
        --gpu0_bsz 0 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ./data/enwik8/ \
        --dataset enwik8 \
        --seq_len 128 \
        --attn_span 768 \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
