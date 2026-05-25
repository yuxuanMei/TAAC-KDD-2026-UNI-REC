"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import (
    FeatureSchema,
    get_pcvr_data,
    NUM_TIME_BUCKETS,
    DEFAULT_TIME_ZONE_OFFSET_HOURS,
    DEFAULT_OOV_MAX_VOCAB_SIZE,
    OOV_UNK_MAP_FILENAME,
    save_oov_unk_maps,
)
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N percent)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--full_train', action='store_true', default=False,
                        help='Use all Row Groups for training and disable '
                             'validation/early stopping. The latest epoch is '
                             'saved as the platform-visible best checkpoint.')
    parser.add_argument('--full_train_epochs', type=int, default=12,
                        help='Epoch count used by --full_train when '
                             '--num_epochs is left at its default 999.')
    parser.add_argument('--full_train_keep_all_ckpts', action='store_true',
                        default=False,
                        help='In --full_train mode, keep every epoch checkpoint '
                             'as a separate platform-visible best_model dir.')
    parser.add_argument('--split_time_probe', action='store_true',
                        default=False,
                        help='Print the startup train/valid timestamp probe. '
                             'Off by default because the split distribution is '
                             'already known and the log is verbose.')
    parser.add_argument('--no_split_time_probe', dest='split_time_probe',
                        action='store_false',
                        help=argparse.SUPPRESS)
    parser.add_argument('--valid_time_slice_metrics',
                        dest='valid_time_slice_metrics',
                        action='store_true', default=True,
                        help='During validation, also log UTC+8 morning-rush '
                             'and test-window AUC/LogLoss diagnostics.')
    parser.add_argument('--no_valid_time_slice_metrics',
                        dest='valid_time_slice_metrics',
                        action='store_false',
                        help='Disable validation time-slice diagnostics.')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')
    parser.add_argument('--seq_time_feature_dim', type=int, default=4, choices=[4, 7, 12],
                        help='Per-step sequence time feature dimension. '
                             '4 = legacy hour/weekday cyclic features; '
                             '7 = add log-age, same-day, and hour-phase '
                             'similarity; '
                             '12 = add age, same-hour/day, and target-hour signals.')
    parser.add_argument('--seq_time_features7', dest='seq_time_feature_dim',
                        action='store_const', const=7,
                        help='Shortcut for --seq_time_feature_dim 7.')
    parser.add_argument('--seq_time_features12', dest='seq_time_feature_dim',
                        action='store_const', const=12,
                        help='Shortcut for --seq_time_feature_dim 12.')
    parser.add_argument('--history_match_dense', action='store_true', default=False,
                        help='Enable 64 deterministic item-history overlap features '
                             'in the otherwise-empty item_dense side.')
    parser.add_argument('--use_predict_rule_calibrator', action='store_true',
                        default=False,
                        help='Apply a tiny profile-based logit calibrator only '
                             'inside model.predict() for inference.')
    parser.add_argument('--no_predict_rule_calibrator',
                        dest='use_predict_rule_calibrator',
                        action='store_false',
                        help='Disable --use_predict_rule_calibrator.')
    parser.add_argument('--predict_rule_scale', type=float, default=0.02,
                        help='Maximum absolute logit delta used by the '
                             'predict-time rule calibrator.')
    parser.add_argument('--use_semantic_rule_features', action='store_true',
                        default=False,
                        help='Convert the P/N semantic rerank rules into '
                             'train-time model features and a learnable '
                             'rule logit residual.')
    parser.add_argument('--semantic_rule_direct_scale', type=float, default=1.0,
                        help='Scale for the learnable semantic-rule logit '
                             'residual when --use_semantic_rule_features is on.')
    parser.add_argument('--semantic_rule_weight_alpha', type=float, default=0.0,
                        help='Extra BCE/Focal sample weight for rows matching '
                             'semantic P/N rules. 0 disables weighting.')
    parser.add_argument('--semantic_rule_pair_alpha', type=float, default=0.0,
                        help='Small in-batch pairwise ranking loss that pushes '
                             'P-rule samples above N-rule samples. 0 disables it.')
    parser.add_argument('--use_dense_int_pair', '--dense_int_pair',
                        dest='use_dense_int_pair', action='store_true',
                        default=False,
                        help='Use aligned user_dense stats (fid 62~66) to gate '
                             'user/item int NS tokens. Pretrained dense fid 61/87 '
                             'and rank vectors 89/90/91 are excluded.')
    parser.add_argument('--time_zone_offset_hours', type=float,
                        default=DEFAULT_TIME_ZONE_OFFSET_HOURS,
                        help='Wall-clock offset for hour/weekday sequence time '
                             'features. Default 8.0 maps UTC timestamps to '
                             'Beijing time.')
    parser.add_argument('--utc_time_features', dest='time_zone_offset_hours',
                        action='store_const', const=0.0,
                        help='Legacy mode: keep hour/weekday features in UTC.')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Use torch.compile for faster training (requires PyTorch 2.0+)')
    parser.add_argument('--use_dense_swa', action='store_true', default=False,
                        help='After training, overwrite the best checkpoint with '
                             'dense-only SWA weights. Embedding tables remain from '
                             'the best EMA checkpoint.')
    parser.add_argument('--dense_swa_top_k', type=int, default=3,
                        help='Number of top validation snapshots used by '
                             '--use_dense_swa.')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--label_smoothing', type=float, default=0.01,
                        help='Label smoothing epsilon (0 = disabled). '
                             'Smooths binary labels: y = y*(1-eps) + (1-y)*eps')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')
    parser.add_argument('--use_oov_unk', action='store_true', default=False,
                        help='Map train-rare or train-unseen sparse IDs to a '
                             'dedicated per-feature UNK row instead of using '
                             'randomly initialized online-only embeddings.')
    parser.add_argument('--oov_min_count', type=int, default=2,
                        help='Minimum training frequency required to keep a '
                             'sparse ID. Lower-frequency IDs map to UNK when '
                             '--use_oov_unk is enabled.')
    parser.add_argument('--oov_max_vocab_size', type=int, default=0,
                        help='Largest vocab_size to scan for --use_oov_unk. '
                             '0 = use --emb_skip_threshold when set, otherwise '
                             f'{DEFAULT_OOV_MAX_VOCAB_SIZE}.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')
    if args.full_train and args.num_epochs == 999:
        args.num_epochs = args.full_train_epochs

    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    oov_max_vocab_size = args.oov_max_vocab_size
    if oov_max_vocab_size <= 0:
        oov_max_vocab_size = (
            args.emb_skip_threshold if args.emb_skip_threshold > 0
            else DEFAULT_OOV_MAX_VOCAB_SIZE
        )
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
        seq_time_feature_dim=args.seq_time_feature_dim,
        history_match_dense=args.history_match_dense,
        time_zone_offset_hours=args.time_zone_offset_hours,
        split_time_probe=args.split_time_probe,
        use_oov_unk=args.use_oov_unk,
        oov_min_count=args.oov_min_count,
        oov_max_vocab_size=oov_max_vocab_size,
        full_train=args.full_train,
    )
    args.oov_unk_map_path = None
    args.oov_max_vocab_size_resolved = oov_max_vocab_size
    if args.use_oov_unk:
        if pcvr_dataset.oov_unk_maps:
            args.oov_unk_map_path = os.path.join(
                args.ckpt_dir, OOV_UNK_MAP_FILENAME)
            save_oov_unk_maps(
                pcvr_dataset.oov_unk_maps,
                args.oov_unk_map_path,
                pcvr_dataset.oov_unk_meta,
            )
        else:
            logging.warning(
                "--use_oov_unk was set, but no OOV/UNK maps were built.")

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    def _item_fids_info(fids):
        info = []
        for fid in fids:
            try:
                offset, length = pcvr_dataset.item_int_schema.get_offset_length(fid)
            except KeyError:
                continue
            vs = max(pcvr_dataset.item_int_vocab_sizes[offset:offset + length])
            info.append((int(fid), int(offset), int(length), int(vs)))
        return info

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "user_dense_fids_info": [
            (fid, offset, length)
            for fid, offset, length in pcvr_dataset.user_dense_schema.entries
        ],
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "seq_fid_lists": {
            domain: list(pcvr_dataset.sideinfo_fids.get(domain, []))
            for domain in pcvr_dataset.seq_domains
        },
        "item_rule_fids_info": _item_fids_info([5, 6, 7, 8, 10, 12, 16, 83, 84, 85]),
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "seq_time_feature_dim": args.seq_time_feature_dim,
        "use_dense_int_pair": args.use_dense_int_pair,
        "use_predict_rule_calibrator": args.use_predict_rule_calibrator,
        "predict_rule_scale": args.predict_rule_scale,
        "time_zone_offset_hours": args.time_zone_offset_hours,
        "use_semantic_rule_features": args.use_semantic_rule_features,
        "semantic_rule_direct_scale": args.semantic_rule_direct_scale,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
    }

    model = PCVRHyFormer(**model_args).to(args.device)

    if args.compile:
        logging.info("Compiling model with torch.compile (reduce-overhead)...")
        # dynamic=True prevents recompilations on dynamic batch sizes and seq lengths
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        use_dense_swa=args.use_dense_swa,
        dense_swa_top_k=args.dense_swa_top_k,
        valid_time_slice_metrics=args.valid_time_slice_metrics,
        oov_unk_map_path=args.oov_unk_map_path,
        full_train=args.full_train,
        full_train_keep_all_ckpts=args.full_train_keep_all_ckpts,
        semantic_rule_weight_alpha=args.semantic_rule_weight_alpha,
        semantic_rule_pair_alpha=args.semantic_rule_pair_alpha,
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
