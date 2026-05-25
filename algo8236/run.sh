#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: RankMixer NS tokenizer ----
# Generalization-focused: baseline dimensions for speed + higher regularization
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --d_model 64 \
    --num_heads 4 \
    --num_hyformer_blocks 2 \
    --batch_size 256 \
    --dropout_rate 0.1 \
    --reinit_sparse_after_epoch 3 \
    --reinit_cardinality_threshold 10000 \
    --sparse_weight_decay 1e-5 \
    --use_oov_unk \
    --oov_min_count 2 \
    --use_semantic_rule_features \
    --semantic_rule_weight_alpha 0.08 \
    --semantic_rule_pair_alpha 0.01 \
    --use_predict_rule_calibrator \
    --predict_rule_scale 0.02 \
    --full_train \
    --num_epochs 11 \
    --full_train_keep_all_ckpts \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
