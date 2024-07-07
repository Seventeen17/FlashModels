#!/bin/bash
set -ex

# FSDP
# note: this need transformers>=4.41.0
# XLA_HLO_DEBUG_VERBOSE_STACK=1 USE_TORCHACC=1 XLA_DUMP_FATAL_STACK=1 XLA_DUMP_HLO_GRAPH=1 XLA_SAVE_TENSORS_FILE=XLA_SAVE_TENSORS_FILE.txt XLA_SAVE_HLO_FILE=XLA_SAVE_HLO_FILE.txt XLA_SAVE_TENSORS_FMT=hlo XLA_METRICS_FILE=XLA_METRICS_FILE.txt XLA_IR_DEBUG=1 PT_XLA_DEBUG=1 XLA_HLO_DEBUG=1 XLA_SYNC_WAIT=1 XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=./hlo/llama3_alpaca_1b_longest_4bucket_2" \

CUDA_VISIBLE_DEVICES=7 ./examples/run.sh --model ./hf_models/config/llama-3-1b --accelerator acc --gc --mbs 8 --fsdp 1 --max_seq_length 512 --use_flash_attn 2>&1 | tee llama3_alpaca_1b_maxlength512.log
