#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python3 train.py \
        --cuda \
        --data ./data/wikitext-103/ \
        --dataset wt103 \
        --adaptive \
        --n_layer 16 \
        --d_model 410 \
        --n_head 10 \
        --d_head 41 \
        --d_inner 2100 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 200000 \
        --seq_len 150 \
        --attn_span 150 \
        --eval_seq_len 150 \
        --batch_size 60 \
        --multi_gpu \
        --gpu0_bsz 4 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python3 eval.py \
        --cuda \
        --data ./data/wikitext-103/ \
        --dataset wt103 \
        --seq_len 64 \
        --attn_span 150 \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
