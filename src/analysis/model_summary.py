def print_model_summary(model):
    print("=== Model Summary ===")
    total, trainable = 0, 0
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            print(f"  {name:<40} {module.__class__.__name__:<20} params={params:,}")
        total += sum(p.numel() for p in module.parameters(recurse=False))
        trainable += sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
    print(f"\n  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
