"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import copy
import glob
import shutil
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader],
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        use_dense_swa: bool = False,
        dense_swa_top_k: int = 3,
        valid_time_slice_metrics: bool = True,
        oov_unk_map_path: Optional[str] = None,
        full_train: bool = False,
        full_train_keep_all_ckpts: bool = False,
        semantic_rule_weight_alpha: float = 0.0,
        semantic_rule_pair_alpha: float = 0.0,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: Optional[DataLoader] = valid_loader
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98)
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.label_smoothing: float = label_smoothing
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.oov_unk_map_path: Optional[str] = oov_unk_map_path
        self.full_train: bool = bool(full_train)
        self.full_train_keep_all_ckpts: bool = bool(full_train_keep_all_ckpts)
        self.semantic_rule_weight_alpha: float = max(0.0, float(semantic_rule_weight_alpha))
        self.semantic_rule_pair_alpha: float = max(0.0, float(semantic_rule_pair_alpha))
        self.use_dense_swa: bool = bool(use_dense_swa)
        self.dense_swa_top_k: int = max(1, int(dense_swa_top_k))
        self.valid_time_slice_metrics: bool = bool(valid_time_slice_metrics)
        self._dense_swa_keys: Optional[List[str]] = None
        self._dense_swa_records: List[Dict[str, Any]] = []

        logging.info(
            "Extra checkpoint export disabled; "
            "training only writes the best checkpoint under TRAIN_CKPT_PATH."
        )
        if self.use_dense_swa:
            logging.info(
                "Dense-only SWA enabled: keep top "
                f"{self.dense_swa_top_k} validation snapshots in memory; "
                "embedding tables stay from the best checkpoint."
            )
        else:
            logging.info("Dense-only SWA disabled; best checkpoint uses EMA weights directly.")

        # EMA (Exponential Moving Average) model for stable evaluation
        self.ema_decay: float = 0.999
        self.ema_model: nn.Module = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        logging.info(f"EMA model created with decay={self.ema_decay}")

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"label_smoothing={label_smoothing}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}, "
                     f"valid_time_slice_metrics={self.valid_time_slice_metrics}, "
                     f"full_train={self.full_train}, "
                     f"full_train_keep_all_ckpts={self.full_train_keep_all_ckpts}, "
                     f"semantic_rule_weight_alpha={self.semantic_rule_weight_alpha}, "
                     f"semantic_rule_pair_alpha={self.semantic_rule_pair_alpha}")

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.layer=2.head=4.hidden=64[.best_model]``.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to three files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        oov_unk_copied = False
        if self.oov_unk_map_path and os.path.exists(self.oov_unk_map_path):
            shutil.copy2(self.oov_unk_map_path, ckpt_dir)
            oov_unk_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            if oov_unk_copied:
                if cfg_to_dump is self.train_config:
                    cfg_to_dump = dict(self.train_config)
                cfg_to_dump['oov_unk_map_path'] = os.path.basename(
                    self.oov_unk_map_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.ema_model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories so that only the latest
        best checkpoint is kept on disk.
        """
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _init_dense_swa_keys(self) -> None:
        """Find state_dict keys that belong to non-Embedding floating tensors."""
        embedding_keys = {
            f"{name}.weight"
            for name, module in self.ema_model.named_modules()
            if isinstance(module, nn.Embedding) and name
        }
        self._dense_swa_keys = [
            key
            for key, value in self.ema_model.state_dict().items()
            if key not in embedding_keys and torch.is_floating_point(value)
        ]
        logging.info(
            "Dense-only SWA key split: "
            f"dense={len(self._dense_swa_keys)}, embedding_kept={len(embedding_keys)}"
        )

    def _record_dense_swa_snapshot(self, val_auc: float, total_step: int) -> None:
        """Keep only the best validation snapshots needed for dense-only SWA."""
        if not self.use_dense_swa:
            return
        if self._dense_swa_keys is None:
            self._init_dense_swa_keys()
        assert self._dense_swa_keys is not None

        state = self.model.state_dict()
        dense_state = {
            key: state[key].detach().cpu().clone()
            for key in self._dense_swa_keys
        }
        self._dense_swa_records.append({
            "auc": float(val_auc),
            "step": int(total_step),
            "state": dense_state,
        })
        self._dense_swa_records.sort(key=lambda r: r["auc"], reverse=True)
        if len(self._dense_swa_records) > self.dense_swa_top_k:
            del self._dense_swa_records[self.dense_swa_top_k:]

        kept = [(round(r["auc"], 6), r["step"]) for r in self._dense_swa_records]
        logging.info(f"Dense-only SWA snapshots kept: {kept}")

    def _finalize_dense_swa_checkpoint(self) -> None:
        """Overwrite the best checkpoint with raw-model dense averaged weights."""
        if not self.use_dense_swa:
            return
        if len(self._dense_swa_records) < 2:
            logging.info(
                "Dense-only SWA skipped: fewer than 2 validation snapshots were kept."
            )
            return

        selected = self._dense_swa_records[:self.dense_swa_top_k]
        best_path = self.early_stopping.checkpoint_path
        if not best_path or not os.path.exists(best_path):
            best_step = int(selected[0]["step"])
            fallback_dir = os.path.join(
                self.save_dir,
                self._build_step_dir_name(best_step, is_best=True),
            )
            fallback_path = os.path.join(fallback_dir, "model.pt")
            if os.path.exists(fallback_path):
                best_path = fallback_path
                self.early_stopping.checkpoint_path = fallback_path
                logging.info(
                    "Dense-only SWA recovered best checkpoint path from "
                    f"top validation step: {best_path}"
                )
            else:
                pattern = os.path.join(
                    self.save_dir,
                    f"global_step{best_step}*.best_model",
                    "model.pt",
                )
                matches = sorted(glob.glob(pattern))
                if matches:
                    best_path = matches[0]
                    self.early_stopping.checkpoint_path = best_path
                    logging.info(
                        "Dense-only SWA recovered best checkpoint path by "
                        f"glob: {best_path}"
                    )

        if not best_path or not os.path.exists(best_path):
            logging.warning(
                f"Dense-only SWA skipped: best checkpoint not found at {best_path!r}."
            )
            return
        if self._dense_swa_keys is None:
            self._init_dense_swa_keys()
        assert self._dense_swa_keys is not None

        full_state = torch.load(best_path, map_location="cpu")
        with torch.no_grad():
            for key in self._dense_swa_keys:
                if key not in full_state:
                    continue
                tensors = [record["state"][key].float() for record in selected]
                avg = torch.stack(tensors, dim=0).mean(dim=0)
                full_state[key] = avg.to(dtype=full_state[key].dtype)

        torch.save(full_state, best_path)

        meta = {
            "method": "dense_only_swa",
            "source": "raw_model",
            "top_k": len(selected),
            "selected": [
                {"auc": record["auc"], "step": record["step"]}
                for record in selected
            ],
            "dense_key_count": len(self._dense_swa_keys),
            "best_checkpoint": os.path.basename(best_path),
        }
        try:
            import json
            with open(os.path.join(os.path.dirname(best_path), "dense_swa_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as exc:
            logging.warning(f"Failed to write dense_swa_meta.json: {exc}")

        logging.info(
            "Dense-only SWA finalized: overwrote best model.pt with averaged "
            f"raw-model dense weights from {len(selected)} validation snapshots; "
            "embedding tables kept from the best checkpoint."
        )

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Persist a new-best checkpoint atomically.

        Flow (ordered to avoid leaving empty sidecar-only directories on disk):

        1. Decide whether ``val_auc`` is *likely* to beat the current best
           using the same threshold as ``EarlyStopping._is_not_improved``,
           so our pre-cleanup and EarlyStopping's internal save decision
           stay in sync.
        2. If unlikely, short-circuit: do nothing on disk. We must NOT
           touch ``self.early_stopping.checkpoint_path`` or call
           ``_write_sidecar_files`` because the target directory may not
           exist yet (sidecar-only dirs would otherwise be created here,
           producing checkpoints with missing ``model.pt``).
        3. If likely, point ``EarlyStopping`` at the canonical
           ``global_stepN.best_model/model.pt`` path, remove any stale
           ``*.best_model`` dirs, then run ``EarlyStopping`` (which writes
           ``model.pt`` when it actually confirms a new best).
        4. Only after ``EarlyStopping`` has confirmed a new best
           (``best_score != old_best``) do we write the sidecar files into
           the freshly-created directory; this is guarded so that a
           razor-close score that tripped ``is_likely_new_best`` but not
           ``EarlyStopping``'s own gate does not create a stray dir.
        """
        self._record_dense_swa_snapshot(val_auc, total_step)

        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            # No new best anticipated: leave disk untouched. The previous
            # best_model dir (with its model.pt + sidecars) remains valid.
            self.early_stopping(val_auc, self.ema_model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })
            return

        # Point EarlyStopping at the canonical best-model location for this
        # step. Only done on the likely-new-best branch so that a skipped
        # save never leaks the unused path into EarlyStopping state.
        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        # Remove stale best dirs first so EarlyStopping's write is the only
        # I/O needed when a new best is confirmed.
        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.ema_model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })

        # Write sidecar files only when EarlyStopping actually confirmed a
        # new best and wrote model.pt. If the score tripped our heuristic
        # but EarlyStopping internally declined to save, skip to avoid
        # creating an empty (sidecar-only) checkpoint directory.
        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True)

    def _save_full_train_checkpoint(
        self,
        total_step: int,
        epoch: int,
        avg_loss: float,
    ) -> None:
        """Save the latest full-train EMA checkpoint in the normal best dir."""
        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")
        if not self.full_train_keep_all_ckpts:
            self._remove_old_best_dirs()
        ckpt_dir = self._save_step_checkpoint(total_step, is_best=True)
        self.early_stopping.best_score = float(epoch)
        self.early_stopping.best_extra_metrics = {
            "full_train_epoch": int(epoch),
            "full_train_avg_loss": float(avg_loss),
        }
        logging.info(
            "Full-train checkpoint saved after epoch "
            f"{epoch}: avg_loss={avg_loss}, path={ckpt_dir}/model.pt")

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.
        """
        print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True)
            loss_sum = 0.0

            for step, batch in train_pbar:
                loss = self._train_step(batch)
                total_step += 1
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)

                train_pbar.set_postfix({"loss": f"{loss:.4f}"})

                # Step-level validation (only when eval_every_n_steps > 0).
                if (
                    not self.full_train
                    and self.eval_every_n_steps > 0
                    and total_step % self.eval_every_n_steps == 0
                ):
                    logging.info(f"Evaluating at step {total_step}")
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                    self._handle_validation_result(total_step, val_auc, val_logloss)

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        self._finalize_dense_swa_checkpoint()
                        return

            avg_loss = loss_sum / len(self.train_loader)
            logging.info(f"Epoch {epoch}, Average Loss: {avg_loss}")

            if self.full_train:
                self._save_full_train_checkpoint(total_step, epoch, avg_loss)
            else:
                val_auc, val_logloss = self.evaluate(epoch=epoch)
                self.model.train()
                torch.cuda.empty_cache()

                logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

                if self.writer:
                    self.writer.add_scalar('AUC/valid', val_auc, total_step)
                    self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                self._handle_validation_result(total_step, val_auc, val_logloss)

                if self.early_stopping.early_stop:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break

            # After the configured epoch, reinitialize high-cardinality sparse
            # params (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

        self._finalize_dense_swa_checkpoint()

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        seq_cyclic_time: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            # Cyclic time features (hour/weekday sin/cos)
            if f'{domain}_cyclic_time' in device_batch:
                seq_cyclic_time[domain] = device_batch[f'{domain}_cyclic_time']
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
            seq_cyclic_time=seq_cyclic_time if seq_cyclic_time else None,
            action_type=None,
            timestamp=device_batch.get('timestamp'),
        )

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)
        logits = self.model(model_input)  # (B, 1)
        logits = logits.squeeze(-1)  # (B,)

        # Classification loss (with optional label smoothing)
        if self.label_smoothing > 0:
            smooth_label = label * (1.0 - self.label_smoothing) + (1.0 - label) * self.label_smoothing
        else:
            smooth_label = label

        if self.loss_type == 'focal':
            per_sample_loss = sigmoid_focal_loss(
                logits,
                smooth_label,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction='none',
            )
        else:
            per_sample_loss = F.binary_cross_entropy_with_logits(
                logits,
                smooth_label,
                reduction='none',
            )

        rule_signed = None
        if (
            self.semantic_rule_weight_alpha > 0.0
            and hasattr(self.model, 'semantic_rule_signed_score')
        ):
            with torch.no_grad():
                rule_signed = self.model.semantic_rule_signed_score(model_input)
                rule_weight = 1.0 + self.semantic_rule_weight_alpha * torch.clamp(
                    rule_signed.abs() * 2.0,
                    min=0.0,
                    max=1.0,
                )
            loss_cls = (per_sample_loss * rule_weight).sum() / rule_weight.sum().clamp_min(1.0)
        else:
            loss_cls = per_sample_loss.mean()

        loss = loss_cls

        if (
            self.semantic_rule_pair_alpha > 0.0
            and hasattr(self.model, 'semantic_rule_signed_score')
        ):
            if rule_signed is None:
                rule_signed = self.model.semantic_rule_signed_score(model_input).detach()
            pos_mask = rule_signed > 0.0
            neg_mask = rule_signed < 0.0
            if bool(pos_mask.any().item()) and bool(neg_mask.any().item()):
                pos_scores = logits[pos_mask]
                neg_scores = logits[neg_mask]
                pair_loss = -F.logsigmoid(
                    pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
                ).mean()
                loss = loss + self.semantic_rule_pair_alpha * pair_loss

        loss.backward()
        # foreach=False: avoids a PyTorch _foreach_norm CUDA kernel bug observed
        # with certain tensor shapes in this project.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

        # Update EMA model
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(model_p.data, alpha=1.0 - self.ema_decay)

        return loss.item()

    def _validation_time_offset_hours(self) -> float:
        """Return the wall-clock offset used by validation time diagnostics."""
        if not self.train_config:
            return 8.0
        try:
            return float(self.train_config.get('time_zone_offset_hours', 8.0))
        except (TypeError, ValueError):
            return 8.0

    def _log_valid_time_slice_metrics(
        self,
        epoch: int,
        logits: torch.Tensor,
        labels: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> None:
        """Log AUC/LogLoss on validation slices that mimic online test time."""
        if not self.valid_time_slice_metrics:
            return
        if timestamps.numel() == 0:
            return

        offset_hours = self._validation_time_offset_hours()
        labels_np = labels.numpy()
        probs_np = torch.sigmoid(logits).numpy()
        ts_np = timestamps.numpy().astype(np.float64)
        wall_hour = np.remainder(ts_np + offset_hours * 3600.0, 86400.0) / 3600.0

        def _slice_line(name: str, mask: np.ndarray) -> None:
            n = int(mask.sum())
            if n == 0:
                logging.info(
                    f"[ValidTimeSlice] epoch={epoch} {name}: rows=0")
                return

            slice_labels = labels_np[mask]
            slice_probs = probs_np[mask]
            mask_t = torch.from_numpy(mask)
            pos_rate = float(slice_labels.mean())
            if len(np.unique(slice_labels)) >= 2:
                auc_msg = f"{float(roc_auc_score(slice_labels, slice_probs)):.10f}"
            else:
                auc_msg = "nan"
            logloss = F.binary_cross_entropy_with_logits(
                logits[mask_t], labels[mask_t].float()).item()
            logging.info(
                "[ValidTimeSlice] "
                f"epoch={epoch} offset_hours={offset_hours:g} "
                f"{name}: rows={n}, pos_rate={pos_rate:.6f}, "
                f"AUC={auc_msg}, LogLoss={logloss:.10f}"
            )

        _slice_line("full", np.ones_like(wall_hour, dtype=bool))
        _slice_line("rush_7_10", (wall_hour >= 7.0) & (wall_hour < 10.0))
        _slice_line(
            "test_window_0740_0915",
            (wall_hour >= (7.0 + 40.0 / 60.0))
            & (wall_hour < (9.0 + 15.0 / 60.0)),
        )
        _slice_line("non_rush", (wall_hour < 7.0) | (wall_hour >= 10.0))

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation over ``self.valid_loader`` and return ``(AUC, logloss)``.

        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing both metrics.
        """
        print("Start Evaluation (PCVRHyFormer) - validation [EMA]")
        if self.valid_loader is None:
            raise RuntimeError("evaluate() called in full_train mode without a valid_loader")
        self.ema_model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_logits_list = []
        all_labels_list = []
        all_timestamps_list = []

        with torch.no_grad():
            for step, batch in pbar:
                logits, labels, timestamps = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())
                all_timestamps_list.append(timestamps.detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()
        all_timestamps = torch.cat(all_timestamps_list, dim=0).long()

        # Filter NaN predictions (may appear if gradients explode).
        nan_mask = torch.isnan(all_logits)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(
                f"[Evaluate] {n_nan}/{len(all_logits)} predictions are NaN, "
                "filtering them out")

        valid_mask = ~nan_mask
        valid_logits = all_logits[valid_mask]
        valid_labels = all_labels[valid_mask]
        valid_timestamps = all_timestamps[valid_mask]

        # Binary AUC via sklearn.
        probs = torch.sigmoid(valid_logits).numpy()
        labels_np = valid_labels.numpy()

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        # Binary logloss (same NaN filtering).
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        self._log_valid_time_slice_metrics(
            epoch, valid_logits, valid_labels, valid_timestamps)

        return auc, logloss

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one validation step and return ``(logits, labels, timestamps)``."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']
        timestamp = device_batch['timestamp']

        model_input = self._make_model_input(device_batch)
        logits = self.ema_model.predict(model_input)  # (B, 1)
        logits = logits.squeeze(-1)  # (B,)

        return logits, label, timestamp
