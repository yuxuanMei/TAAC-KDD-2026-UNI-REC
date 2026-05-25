#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: semantic token redistribution lite ----
# Field-aware tokenizer keeps each fid embedding intact, then reallocates
# token capacity from user-heavy old V9 (5/2) toward item/ads (3/4).
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type fieldaware \
    --user_ns_tokens 3 \
    --item_ns_tokens 4 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --d_model 64 \
    --num_heads 4 \
    --num_hyformer_blocks 2 \
    --batch_size 256 \
    --dropout_rate 0.1 \
    --loss_type focal \
    --focal_alpha 0.75 \
    --focal_gamma 1.5 \
    --label_smoothing 0 \
    --reinit_sparse_after_epoch 3 \
    --reinit_cardinality_threshold 10000 \
    --sparse_weight_decay 1e-5 \
    --use_oov_unk \
    --oov_min_count 2 \
    --rare16_profile \
    --use_dense_int_pair \
    --use_abs_time_ns \
    --use_session_crossday_time \
    --full_train \
    --num_epochs 15 \
    --full_train_keep_all_ckpts \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.

# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     --d_model 64 \
#     --num_heads 4 \
#     --num_hyformer_blocks 2 \
#     --batch_size 256 \
#     --dropout_rate 0.1 \
#     --reinit_sparse_after_epoch 3 \
#     --reinit_cardinality_threshold 10000 \
#     --sparse_weight_decay 1e-5 \
#     --use_oov_unk \
#     --oov_min_count 2 \
#     --rare16_profile \
#     --use_oov_residual_calibrator \
#     --oov_residual_scale 0.05 \
#     --full_train \
#     --num_epochs 15 \
#     --full_train_keep_all_ckpts \
#     "$@"
