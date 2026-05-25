import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import wandb
import sklearn.metrics
from torch.amp import GradScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.data import get_loaders, get_fusion_loaders
from src.networks.dcn import DCNv2
from src.networks.fusion import GatedFusionModel, get_tokenizer, precompute_bert_embeddings
from src.analysis.model_summary import print_model_summary
from src.analysis.evaluation import tune_thresholds
from src.analysis.visualization import plot_roc_curve, plot_precision_recall_curve
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


def run_once(args, seed: int, run) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.use_semantic:
        from src.cached_dataset import get_cached_loaders
        
        # Detect if running on Modal (cache mounted at /app/cache)
        if os.path.exists('/app/cache'):
            cache_dir = '/app/cache'
        else:
            cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        
        train_cache = os.path.join(cache_dir, 'train_cached.pt')
        
        if not os.path.exists(train_cache):
            raise FileNotFoundError(f"Cached embeddings not found at {train_cache}. Run precompute_embeddings.py first.")
        
        train_loader, val_loader, test_loader = get_cached_loaders(
            cache_dir, batch_size=args.batch_size, num_workers=args.num_workers)
        
        sample_features, sample_embeds, _ = next(iter(train_loader))
        input_dim = sample_features.shape[1]
        bert_dim = sample_embeds.shape[1]
        
        model = GatedFusionModel(input_dim=input_dim, embed_dim=args.embed_dim,
                                 cross_type=args.cross_type, deep_type=args.deep_type).to(device)
    else:
        train_loader, val_loader, test_loader, pos_weight = get_loaders(
            batch_size=args.batch_size, num_workers=args.num_workers, tiny=args.tiny)
        input_dim = next(iter(train_loader))[0].shape[1]
        model = DCNv2(input_dim=input_dim, cross_type=args.cross_type, deep_type=args.deep_type).to(device)

    if seed == args.seed[0]:
        print_model_summary(model)

    if args.use_semantic:
        _, _, labels_batch = next(iter(train_loader))
        y_train = labels_batch.numpy()
        pos_weight = torch.tensor((1 - y_train).sum(axis=0) / (y_train.sum(axis=0) + 1e-6), dtype=torch.float32)
        pos_weight = torch.sqrt(pos_weight)

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
    ckpt = f'artifacts/best_{"fusion" if args.use_semantic else "model"}_seed{seed}.pt'
    os.makedirs('artifacts', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1_micro, train_f1_macro, train_auc_micro, train_auc_macro = train_epoch(
            model, train_loader, optimizer, criterion, device, args.use_semantic, scaler)
        val_loss, val_acc, val_f1_micro, val_f1_macro, val_auc_micro, val_auc_macro, _, _ = eval_epoch(
            model, val_loader, criterion, device, args.use_semantic)
        scheduler.step(val_loss)
        
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/acc': train_acc,
            'train/f1_micro': train_f1_micro,
            'train/f1_macro': train_f1_macro,
            'train/auc_micro': train_auc_micro,
            'train/auc_macro': train_auc_macro,
            'val/loss': val_loss,
            'val/acc': val_acc,
            'val/f1_micro': val_f1_micro,
            'val/f1_macro': val_f1_macro,
            'val/auc_micro': val_auc_micro,
            'val/auc_macro': val_auc_macro,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        if val_auc_macro > best_val_auc:
            best_val_auc = val_auc_macro
            torch.save(model.state_dict(), ckpt)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                break

    model.load_state_dict(torch.load(ckpt))
    test_loss, test_acc, test_f1_micro, test_f1_macro, test_auc_micro, test_auc_macro, test_probs, test_labels = eval_epoch(
        model, test_loader, criterion, device, args.use_semantic)
    
    _, _, _, _, _, _, val_probs, val_labels = eval_epoch(model, val_loader, criterion, device, args.use_semantic)
    thresholds = tune_thresholds(val_probs, val_labels)
    
    wandb.log({
        'test/loss': test_loss,
        'test/acc': test_acc,
        'test/f1_micro': test_f1_micro,
        'test/f1_macro': test_f1_macro,
        'test/auc_micro': test_auc_micro,
        'test/auc_macro': test_auc_macro
    })
    
    import matplotlib.pyplot as plt
    
    for i, label in enumerate(LABEL_COLUMNS):
        preds = (test_probs[:, i] >= thresholds[i]).astype(int)
        cm = sklearn.metrics.confusion_matrix(test_labels[:, i], preds)
        fig, ax = plt.subplots()
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Confusion Matrix - {label}')
        wandb.log({f'test/confusion_matrix_{label}': wandb.Image(fig)})
        plt.close(fig)
    
    fig_roc = plot_roc_curve(test_labels, test_probs, LABEL_COLUMNS)
    wandb.log({'test/roc_curve': wandb.Image(fig_roc)})
    plt.close(fig_roc)
    
    fig_pr = plot_precision_recall_curve(test_labels, test_probs, LABEL_COLUMNS)
    wandb.log({'test/pr_curve': wandb.Image(fig_pr)})
    plt.close(fig_pr)

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
    parser.add_argument('--use_semantic', action='store_true')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--wandb_project', type=str, default='smellycode-dcnv2')
    parser.add_argument('--wandb_entity', type=str, default=None)
    args = parser.parse_args()

    all_results = []
    for seed in args.seed:
        with wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"seed_{seed}",
            reinit=True
        ) as run:
            result = run_once(args, seed, run)
            all_results.append(result)

    if len(args.seed) > 1:
        print("\n=== Multi-seed Summary ===")
        print(f"{'Metric':<20} {'Mean':>8} {'Std':>8}")
        print("-" * 38)
        for m in all_results[0]:
            vals = [r[m] for r in all_results]
            print(f"  {m:<20} {np.mean(vals):>8.3f} {np.std(vals):>8.3f}")


if __name__ == '__main__':
    main()
