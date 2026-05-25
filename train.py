import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.amp import GradScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.data import get_loaders, get_fusion_loaders
from src.networks.dcn import DCNv2
from src.networks.fusion import GatedFusionModel, get_tokenizer, precompute_bert_embeddings
from src.analysis.model_summary import print_model_summary
from src.analysis.evaluation import evaluate_per_label, tune_thresholds
from src.losses.focal_loss import MultilabelFocalLoss, AsymmetricLoss
from config import LABEL_COLUMNS, SEED


def compute_metrics(probs: np.ndarray, labels: np.ndarray):
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    try:
        auc_micro = roc_auc_score(labels, probs, average='micro')
        auc_macro = roc_auc_score(labels, probs, average='macro')
    except ValueError:
        auc_micro = auc_macro = 0.0
    return acc, f1_micro, f1_macro, auc_micro, auc_macro


def _forward(model, batch, device, use_semantic):
    if use_semantic:
        features, bert_embed, labels = batch
        logits = model(features.to(device), bert_embed.to(device))
        return logits, labels.to(device)
    else:
        features, labels = batch
        logits, _ = model(features.to(device))
        return logits, labels.to(device)


def make_cached_dataset(fusion_dataset, bert_embeds: torch.Tensor):
    """Wrap a CodeSmellFusionDataset, replacing code strings with pre-computed embeddings."""
    from torch.utils.data import TensorDataset
    return TensorDataset(fusion_dataset.features, bert_embeds, fusion_dataset.labels)


def train_epoch(model, loader, optimizer, criterion, device, use_semantic, scaler):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in loader:
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_semantic and device.type == 'cuda'):
            logits, labels = _forward(model, batch, device, use_semantic)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return total_loss / len(loader), *compute_metrics(all_preds, all_labels)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, use_semantic):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in loader:
        logits, labels = _forward(model, batch, device, use_semantic)
        total_loss += criterion(logits, labels).item()
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return total_loss / len(loader), *compute_metrics(all_preds, all_labels), all_preds, all_labels


def run_once(args, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Seed {seed}] device={device} | Cross={args.cross_type} | Deep={args.deep_type} | Semantic={args.use_semantic}")

    if args.use_semantic:
        from torch.utils.data import DataLoader, TensorDataset
        tokenizer = get_tokenizer()
        raw_train, raw_val, raw_test, pos_weight = get_fusion_loaders(
            batch_size=args.batch_size, num_workers=args.num_workers, tiny=args.tiny)

        # Build a temp BERT model just for pre-computation
        from transformers import AutoModel
        bert_tmp = AutoModel.from_pretrained(GatedFusionModel.BERT_MODEL).to(device)
        bert_tmp.eval()

        print("Pre-computing BERT embeddings (train)...")
        train_embeds = precompute_bert_embeddings(
            raw_train.dataset.codes, tokenizer, bert_tmp, device, max_length=args.max_length)
        print("Pre-computing BERT embeddings (val)...")
        val_embeds = precompute_bert_embeddings(
            raw_val.dataset.codes, tokenizer, bert_tmp, device, max_length=args.max_length)
        print("Pre-computing BERT embeddings (test)...")
        test_embeds = precompute_bert_embeddings(
            raw_test.dataset.codes, tokenizer, bert_tmp, device, max_length=args.max_length)
        del bert_tmp
        torch.cuda.empty_cache() if device.type == 'cuda' else None

        train_ds = TensorDataset(raw_train.dataset.features, train_embeds, raw_train.dataset.labels)
        val_ds   = TensorDataset(raw_val.dataset.features,   val_embeds,   raw_val.dataset.labels)
        test_ds  = TensorDataset(raw_test.dataset.features,  test_embeds,  raw_test.dataset.labels)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        input_dim = train_ds.tensors[0].shape[1]
        model = GatedFusionModel(input_dim=input_dim, embed_dim=args.embed_dim,
                                 cross_type=args.cross_type, deep_type=args.deep_type).to(device)
    else:
        train_loader, val_loader, test_loader, pos_weight = get_loaders(
            batch_size=args.batch_size, num_workers=args.num_workers, tiny=args.tiny)
        input_dim = next(iter(train_loader))[0].shape[1]
        model = DCNv2(input_dim=input_dim, cross_type=args.cross_type, deep_type=args.deep_type).to(device)

    if seed == args.seed[0]:
        print_model_summary(model)

    if args.loss == 'focal':
        criterion = MultilabelFocalLoss(gamma=args.focal_gamma, alpha=pos_weight.to(device))
    elif args.loss == 'asl':
        criterion = AsymmetricLoss(gamma_neg=args.asl_gamma_neg, gamma_pos=args.asl_gamma_pos, alpha=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = GradScaler('cuda', enabled=args.use_semantic and device.type == 'cuda')

    best_val_auc, patience_counter = 0, 0
    os.makedirs('artifacts', exist_ok=True)
    ckpt = f'artifacts/best_{"fusion" if args.use_semantic else "model"}_seed{seed}.pt'

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1_micro, train_f1_macro, train_auc_micro, train_auc_macro = train_epoch(
            model, train_loader, optimizer, criterion, device, args.use_semantic, scaler)
        val_loss, val_acc, val_f1_micro, val_f1_macro, val_auc_micro, val_auc_macro, _, _ = eval_epoch(
            model, val_loader, criterion, device, args.use_semantic)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:2d} | Train Loss {train_loss:.4f} Acc {train_acc:.3f} F1-micro {train_f1_micro:.3f} "
              f"F1-macro {train_f1_macro:.3f} AUC-macro {train_auc_macro:.3f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.3f} F1-micro {val_f1_micro:.3f} "
              f"F1-macro {val_f1_macro:.3f} AUC-macro {val_auc_macro:.3f}")
        if val_auc_macro > best_val_auc:
            best_val_auc = val_auc_macro
            torch.save(model.state_dict(), ckpt)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(ckpt))
    test_loss, test_acc, test_f1_micro, test_f1_macro, test_auc_micro, test_auc_macro, test_probs, test_labels = eval_epoch(
        model, test_loader, criterion, device, args.use_semantic)
    print(f"Test: Loss {test_loss:.4f} Acc {test_acc:.3f} F1-micro {test_f1_micro:.3f} "
          f"F1-macro {test_f1_macro:.3f} AUC-micro {test_auc_micro:.3f} AUC-macro {test_auc_macro:.3f}")

    _, _, _, _, _, _, val_probs, val_labels = eval_epoch(model, val_loader, criterion, device, args.use_semantic)
    thresholds = tune_thresholds(val_probs, val_labels)
    print("=== Per-label Eval (tuned thresholds) ===")
    evaluate_per_label(test_probs, test_labels, LABEL_COLUMNS, thresholds)

    return {"loss": test_loss, "acc": test_acc, "f1_micro": test_f1_micro, "f1_macro": test_f1_macro,
            "auc_micro": test_auc_micro, "auc_macro": test_auc_macro}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cross_type', choices=['standard', 'gated'], default='standard')
    parser.add_argument('--deep_type', choices=['bottleneck', 'moe'], default='bottleneck')
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, nargs='+', default=[SEED])
    parser.add_argument('--loss', choices=['bce', 'focal', 'asl'], default='bce')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--asl_gamma_neg', type=float, default=4.0)
    parser.add_argument('--asl_gamma_pos', type=float, default=1.0)
    # Semantic fusion args
    parser.add_argument('--use_semantic', action='store_true')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=512)
    args = parser.parse_args()

    all_results = []
    for seed in args.seed:
        result = run_once(args, seed)
        all_results.append(result)
        print(f"\n[Seed {seed}] Test: " + " ".join(f"{k}={v:.3f}" for k, v in result.items()))

    if len(args.seed) > 1:
        print("\n=== Multi-seed Summary ===")
        print(f"{'Metric':<20} {'Mean':>8} {'Std':>8}")
        print("-" * 38)
        for m in all_results[0]:
            vals = [r[m] for r in all_results]
            print(f"  {m:<20} {np.mean(vals):>8.3f} {np.std(vals):>8.3f}")


if __name__ == '__main__':
    main()
