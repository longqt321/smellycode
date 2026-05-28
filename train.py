import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import wandb
import sklearn.metrics
from torch.amp import GradScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from src.data import get_loaders, get_fusion_loaders
from src.networks.dcn import DCNv2
from src.networks.fusion import GatedFusionModel, LateFusionMLPModel, get_tokenizer, precompute_bert_embeddings
from src.analysis.model_summary import print_model_summary
from src.analysis.threshold_optimization import tune_thresholds, tune_thresholds_bayesian, tune_thresholds_roc_based, compare_threshold_methods
from src.analysis.visualization import plot_roc_curve, plot_precision_recall_curve
from src.analysis.gate_visualization import create_gate_report, interpret_gate_behavior
from src.losses.focal_loss import MultilabelFocalLoss, AsymmetricLoss
from src.losses.class_balanced_loss import ClassBalancedFocalLoss, ClassBalancedLoss, get_class_balanced_weights_from_dataloader
from src.utils.onnx_export import export_dcn_to_onnx, export_fusion_to_onnx
from config import LABEL_COLUMNS, SEED


def compute_metrics(probs: np.ndarray, labels: np.ndarray):
    """
    Tính metrics cho multi‑label classification với imbalance nặng.
    - balanced_acc: macro recall (công bằng giữa các lớp).
    - Giữ nguyên cấu trúc output để không phá vỡ pipeline.
    """
    preds = (probs >= 0.5).astype(int)
    # Dùng macro recall làm balanced accuracy cho đa nhãn
    acc = recall_score(labels, preds, average='macro', zero_division=0)
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


def _get_gate_stats(model):
    if hasattr(model, 'gate_stats') and callable(model.gate_stats):
        return model.gate_stats()
    return {}

def make_cached_dataset(fusion_dataset, bert_embeds: torch.Tensor):
    """Wrap a CodeSmellFusionDataset, replacing code strings with pre-computed embeddings."""
    from torch.utils.data import TensorDataset
    return TensorDataset(fusion_dataset.features, bert_embeds, fusion_dataset.labels)


def train_epoch(model, loader, optimizer, criterion, device, use_semantic, scaler):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    gate_stats_accumulator = []
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
        gate_stats = _get_gate_stats(model)
        if gate_stats:
            gate_stats_accumulator.append(gate_stats)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    aggregated_gate_stats = {}
    if gate_stats_accumulator:
        for key in gate_stats_accumulator[0]:
            aggregated_gate_stats[f'train/{key}'] = float(np.mean([s[key] for s in gate_stats_accumulator]))
    return total_loss / len(loader), *compute_metrics(all_preds, all_labels), aggregated_gate_stats


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, use_semantic):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    gate_stats_accumulator = []
    for batch in loader:
        logits, labels = _forward(model, batch, device, use_semantic)
        total_loss += criterion(logits, labels).item()
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        gate_stats = _get_gate_stats(model)
        if gate_stats:
            gate_stats_accumulator.append(gate_stats)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    aggregated_gate_stats = {}
    if gate_stats_accumulator:
        for key in gate_stats_accumulator[0]:
            aggregated_gate_stats[f'eval/{key}'] = float(np.mean([s[key] for s in gate_stats_accumulator]))
    return total_loss / len(loader), *compute_metrics(all_preds, all_labels), all_preds, all_labels, aggregated_gate_stats


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
        # bert_dim = sample_embeds.shape[1]
        
        FusionClass = GatedFusionModel if args.fusion_type == 'gated' else LateFusionMLPModel
        model = FusionClass(
            input_dim=input_dim,
            embed_dim=args.embed_dim,
            cross_type=args.cross_type,
            deep_type=args.deep_type,
            num_classes=len(LABEL_COLUMNS),
        ).to(device)
    else:
        train_loader, val_loader, test_loader, pos_weight = get_loaders(
            batch_size=args.batch_size, num_workers=args.num_workers, tiny=args.tiny)
        input_dim = next(iter(train_loader))[0].shape[1]
        model = DCNv2(
            input_dim=input_dim,
            cross_type=args.cross_type,
            deep_type=args.deep_type,
            num_classes=len(LABEL_COLUMNS),
        ).to(device)

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
    elif args.loss == 'cb_focal':
        criterion = ClassBalancedFocalLoss(beta=args.cb_beta, gamma=args.focal_gamma)
    elif args.loss == 'cb':
        criterion = ClassBalancedLoss(beta=args.cb_beta)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,weight_decay=1e-2
    )

    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1,
        total_iters=20,
    )
    
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=80,
        eta_min=args.lr * 0.1
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[20],
    )
    scaler = GradScaler('cuda', enabled=args.use_semantic and device.type == 'cuda')

    best_val_auc, patience_counter = 0, 0
    ckpt = f'artifacts/best_{"fusion" if args.use_semantic else "model"}_seed{seed}.pt'
    os.makedirs('artifacts', exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1_micro, train_f1_macro, train_auc_micro, train_auc_macro, train_gate_stats = train_epoch(
            model, train_loader, optimizer, criterion, device, args.use_semantic, scaler)
        val_loss, val_acc, val_f1_micro, val_f1_macro, val_auc_micro, val_auc_macro, _, _, val_gate_stats = eval_epoch(
            model, val_loader, criterion, device, args.use_semantic)
        scheduler.step()
        
        log_payload = {
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
        }
        log_payload.update(train_gate_stats)
        log_payload.update(val_gate_stats)
        wandb.log(log_payload)

        gate_console = ""
        if train_gate_stats:
            gate_console = (
                f"\n  [GATE] train_mean: {train_gate_stats.get('train/gate/mean', 0.0):.4f}"
                f" | train_std: {train_gate_stats.get('train/gate/std', 0.0):.4f}"
                f" | val_mean: {val_gate_stats.get('eval/gate/mean', 0.0):.4f}"
                f" | val_std: {val_gate_stats.get('eval/gate/std', 0.0):.4f}"
                f" | near0: {val_gate_stats.get('eval/gate/near_zero', 0.0):.4f}"
                f" | near1: {val_gate_stats.get('eval/gate/near_one', 0.0):.4f}"
            )
        
        print(
            f"Epoch {epoch:02d}/{args.epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}\n"
            f"  [TRAIN] Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1-Micro: {train_f1_micro:.4f} | F1-Macro: {train_f1_macro:.4f} | AUC-Micro: {train_auc_micro:.4f} | AUC-Macro: {train_auc_macro:.4f}\n"
            f"  [VAL]   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1-Micro: {val_f1_micro:.4f} | F1-Macro: {val_f1_macro:.4f} | AUC-Micro: {val_auc_micro:.4f} | AUC-Macro: {val_auc_macro:.4f}\n"
            f"{gate_console}\n"
            f"{'-'*115}"
        )
        
        if val_auc_macro > best_val_auc:
            best_val_auc = val_auc_macro
            torch.save(model.state_dict(), ckpt)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                break

    model.load_state_dict(torch.load(ckpt))
    test_loss, test_acc, test_f1_micro, test_f1_macro, test_auc_micro, test_auc_macro, test_probs, test_labels, test_gate_stats = eval_epoch(
        model, test_loader, criterion, device, args.use_semantic)
    
    _, _, _, _, _, _, val_probs, val_labels, _ = eval_epoch(model, val_loader, criterion, device, args.use_semantic)
    
    # Advanced threshold optimization
    if args.threshold_method == 'bayesian':
        thresholds = tune_thresholds_bayesian(val_probs, val_labels, metric='f1_macro', n_iterations=30)
    elif args.threshold_method == 'roc':
        thresholds = tune_thresholds_roc_based(val_probs, val_labels, criterion='youden')
    else:  # grid
        thresholds = tune_thresholds(val_probs, val_labels)
    
    print(f"\\nThreshold optimization method: {args.threshold_method}")
    print(f"Optimized thresholds: {[f'{t:.3f}' for t in thresholds]}")
    
    test_log_payload = {
        'test/loss': test_loss,
        'test/acc': test_acc,
        'test/f1_micro': test_f1_micro,
        'test/f1_macro': test_f1_macro,
        'test/auc_micro': test_auc_micro,
        'test/auc_macro': test_auc_macro
    }
    test_log_payload.update({k.replace('eval/', 'test/'): v for k, v in test_gate_stats.items()})
    wandb.log(test_log_payload)
    
    print(
            f"  [TEST] Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1-Micro: {test_f1_micro:.4f} | F1-Macro: {test_f1_macro:.4f} | AUC-Micro: {test_auc_micro:.4f} | AUC-Macro: {test_auc_macro:.4f}\n"
            f"{'-'*115}"
        )
    
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
    
    # Gate visualization for fusion models
    if args.gate_analysis and args.use_semantic and args.fusion_type == 'gated':
        gate_report_dir = os.path.join('artifacts', f'gate_analysis_seed{seed}')
        os.makedirs(gate_report_dir, exist_ok=True)
        
        from src.cached_dataset import get_cached_loaders
        _, gate_loader, _ = get_cached_loaders(
            os.path.dirname(train_cache), 
            batch_size=args.batch_size, 
            num_workers=args.num_workers
        )
        
        gate_stats = create_gate_report(model, gate_loader, device, num_samples=500, save_dir=gate_report_dir)
        interpretation = interpret_gate_behavior(gate_stats)
        print(f"\n{interpretation}")
        
        # Log to wandb
        wandb.log({
            'gate/mean': gate_stats['mean'],
            'gate/std': gate_stats['std'],
            'gate/near_zero_pct': gate_stats['near_zero'] * 100,
            'gate/near_one_pct': gate_stats['near_one'] * 100,
            'gate/neutral_pct': gate_stats['neutral'] * 100,
        })
        
        # Upload visualization images
        wandb.save(os.path.join(gate_report_dir, 'gate_distribution.png'))
        wandb.save(os.path.join(gate_report_dir, 'gate_by_class.png'))

    # ONNX export
    if args.export_onnx:
        onnx_dir = os.path.join('artifacts', f'onnx_export_seed{seed}')
        os.makedirs(onnx_dir, exist_ok=True)
        
        # Also prepare to save to Modal volume if running on Modal
        onnx_volume = None
        volume_onnx_dir = f'/artifacts/onnx_export_seed{seed}'
        try:
            from modal import Volume
            onnx_volume = Volume.from_name("smellycode-onnx", create_if_missing=True)
            # Create directory in volume
            onnx_volume.mkdir(volume_onnx_dir.lstrip('/'), parents=True, exist_ok=True)
        except Exception:
            pass  # Not running on Modal or volume not available
        
        if args.use_semantic:
            onnx_path = os.path.join(onnx_dir, 'fusion_model.onnx')
            sample_features, sample_embeds, _ = next(iter(train_loader))
            input_dim = sample_features.shape[1]
            bert_dim = sample_embeds.shape[1]
            
            export_info = export_fusion_to_onnx(
                model=model,
                input_dim=input_dim,
                bert_dim=bert_dim,
                output_path=onnx_path,
                device=device
            )
            print(f"\n[FUSION MODEL ONNX EXPORT]")
            print(f"  Model saved to: {onnx_path}")
            print(f"  Input dim: {input_dim}, BERT dim: {bert_dim}")
            print(f"  Output dim: {export_info['output_dim']}")
            print(f"  Verification: {'PASSED' if export_info['verified'] else 'FAILED'}")
            print(f"  Max PyTorch-ONNX diff: {export_info['max_pytorch_diff']:.2e}")
            
            # Copy to Modal volume if available
            if onnx_volume is not None:
                try:
                    with open(onnx_path, 'rb') as f:
                        onnx_volume.write_file(f'{volume_onnx_dir}/fusion_model.onnx', f.read())
                    print(f"  Saved to Modal volume: {volume_onnx_dir}/fusion_model.onnx")
                except Exception as e:
                    print(f"  Warning: Could not save to Modal volume: {e}")
            
            wandb.log({
                'onnx/exported': True,
                'onnx/max_diff': export_info['max_pytorch_diff'],
                'onnx/verified': export_info['verified']
            })
        else:
            onnx_path = os.path.join(onnx_dir, 'dcn_model.onnx')
            sample_features, _ = next(iter(train_loader))
            input_dim = sample_features.shape[1]
            
            export_info = export_dcn_to_onnx(
                model=model,
                input_dim=input_dim,
                output_path=onnx_path,
                device=device
            )
            print(f"\n[DCN MODEL ONNX EXPORT]")
            print(f"  Model saved to: {onnx_path}")
            print(f"  Input dim: {input_dim}")
            print(f"  Output dim: {export_info['output_dim']}")
            print(f"  Verification: {'PASSED' if export_info['verified'] else 'FAILED'}")
            print(f"  Max PyTorch-ONNX diff: {export_info['max_pytorch_diff']:.2e}")
            
            # Copy to Modal volume if available
            if onnx_volume is not None:
                try:
                    with open(onnx_path, 'rb') as f:
                        onnx_volume.write_file(f'{volume_onnx_dir}/dcn_model.onnx', f.read())
                    print(f"  Saved to Modal volume: {volume_onnx_dir}/dcn_model.onnx")
                except Exception as e:
                    print(f"  Warning: Could not save to Modal volume: {e}")
            
            wandb.log({
                'onnx/exported': True,
                'onnx/max_diff': export_info['max_pytorch_diff'],
                'onnx/verified': export_info['verified']
            })

    return {"loss": test_loss, "acc": test_acc, "f1_micro": test_f1_macro, "f1_macro": test_f1_macro,
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
    parser.add_argument('--loss', choices=['bce', 'focal', 'asl', 'cb_focal', 'cb'], default='bce')
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--asl_gamma_neg', type=float, default=4.0)
    parser.add_argument('--asl_gamma_pos', type=float, default=1.0)
    parser.add_argument('--cb_beta', type=float, default=0.9999)
    parser.add_argument('--threshold_method', choices=['grid', 'bayesian', 'roc'], default='grid')
    parser.add_argument('--gate_analysis', action='store_true', help='Generate gate visualization report')
    parser.add_argument('--export_onnx', action='store_true', help='Export model to ONNX format after training')
    parser.add_argument('--use_semantic', action='store_true')
    parser.add_argument('--fusion_type', choices=['gated', 'late_mlp'], default='gated')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--wandb_project', type=str, default='smellycode-dcnv2')
    parser.add_argument('--wandb_entity', type=str, default=None)
    args = parser.parse_args()

    all_results = []
    for seed in args.seed:
        run_name = f"semantic={args.use_semantic}_fusion={args.fusion_type}_cross={args.cross_type}_deep={args.deep_type}_loss={args.loss}_seed={seed}"

        
        with wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
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
