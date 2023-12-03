#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python3 train.py \
        --cuda \
        --data ./data/wikitext-103/ \
        --dataset wt103 \
        --adaptive \
        --div_val 4 \
        --n_layer 18 \
        --d_model 1024 \
        --n_head 16 \
        --d_head 64 \
        --d_inner 4096 \
        --dropout 0.2 \
        --dropatt 0.2 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 16000 \
        --max_step 4000000 \
        --seq_len 384 \
        --attn_span 384 \
        --eval_seq_len 128 \
        --batch_size 128 \
        --multi_gpu \
        --gpu0_bsz 0 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python3 eval.py \
        --cuda \
        --data ./data/wikitext-103/ \
        --dataset wt103 \
        --seq_len 128 \
        --attn_span 384 \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
