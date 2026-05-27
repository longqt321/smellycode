"""ONNX export utilities for trained models."""
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np


class DCNv2Wrapper(nn.Module):
    """Wrapper for DCNv2 model to ensure single output for ONNX export."""
    
    def __init__(self, dcn_model: nn.Module):
        super().__init__()
        self.dcn = dcn_model
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only logits."""
        logits, _ = self.dcn(features)
        return logits


def export_dcn_to_onnx(
    model: nn.Module,
    input_dim: int,
    output_path: str,
    device: torch.device = None,
    opset_version: int = 14,
    dynamic_axes: bool = True
) -> Dict[str, Any]:
    """
    Export DCNv2 model to ONNX format.
    
    Args:
        model: Trained DCNv2 model
        input_dim: Dimension of input features
        output_path: Path to save ONNX model
        device: Device to run export on
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic batch size
        
    Returns:
        Dictionary with export metadata
    """
    if device is None:
        device = torch.device('cpu')
    
    model.eval()
    
    # Wrap model if needed
    if not isinstance(model, DCNv2Wrapper):
        wrapper = DCNv2Wrapper(model).to(device)
    else:
        wrapper = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, input_dim, device=device)
    
    # Setup dynamic axes
    dynamic_axes_config = None
    if dynamic_axes:
        dynamic_axes_config = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Export
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes_config,
        verbose=False
    )
    
    # Verify export
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Test inference
    ort_session = ort.InferenceSession(output_path)
    ort_inputs = {'input': dummy_input.cpu().numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # Compare with PyTorch
    with torch.no_grad():
        pt_output = wrapper(dummy_input).cpu().numpy()
    
    max_diff = np.abs(ort_output - pt_output).max()
    
    return {
        'model_path': output_path,
        'input_dim': input_dim,
        'output_dim': ort_output.shape[1],
        'opset_version': opset_version,
        'max_pytorch_diff': float(max_diff),
        'verified': max_diff < 1e-4
    }


def export_fusion_to_onnx(
    model: nn.Module,
    input_dim: int,
    bert_dim: int = 768,
    output_path: str = None,
    device: torch.device = None,
    opset_version: int = 14
) -> Dict[str, Any]:
    """
    Export GatedFusionModel to ONNX format.
    
    Note: This exports a version that accepts pre-computed BERT embeddings.
    For end-to-end export including BERT, use export_fusion_end_to_end.
    
    Args:
        model: Trained GatedFusionModel
        input_dim: Dimension of numeric features
        bert_dim: Dimension of BERT embeddings
        output_path: Path to save ONNX model
        device: Device to run export on
        opset_version: ONNX opset version
        
    Returns:
        Dictionary with export metadata
    """
    from src.networks.fusion import GatedFusionModel, LateFusionMLPModel
    
    if device is None:
        device = torch.device('cpu')
    
    if not isinstance(model, (GatedFusionModel, LateFusionMLPModel)):
        raise ValueError("Model must be GatedFusionModel or LateFusionMLPModel")
    
    model.eval()
    model = model.to(device)
    
    # Create dummy inputs
    dummy_features = torch.randn(1, input_dim, device=device)
    dummy_bert = torch.randn(1, bert_dim, device=device)
    
    # Dynamic axes for variable batch size
    dynamic_axes = {
        'features': {0: 'batch_size'},
        'bert_embed': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    
    # Export
    torch.onnx.export(
        model,
        (dummy_features, dummy_bert),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['features', 'bert_embed'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    # Verify export
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # Test inference
    ort_session = ort.InferenceSession(output_path)
    ort_inputs = {
        'features': dummy_features.cpu().numpy(),
        'bert_embed': dummy_bert.cpu().numpy()
    }
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # Compare with PyTorch
    with torch.no_grad():
        pt_output = model(dummy_features, dummy_bert).cpu().numpy()
    
    max_diff = np.abs(ort_output - pt_output).max()
    
    return {
        'model_path': output_path,
        'input_dim': input_dim,
        'bert_dim': bert_dim,
        'output_dim': ort_output.shape[1],
        'opset_version': opset_version,
        'max_pytorch_diff': float(max_diff),
        'verified': max_diff < 1e-4
    }


def export_fusion_end_to_end(
    model: nn.Module,
    tokenizer,
    input_dim: int,
    output_path: str,
    device: torch.device = None,
    max_length: int = 512,
    opset_version: int = 14
) -> Dict[str, Any]:
    """
    Export GatedFusionModel with integrated BERT for end-to-end inference.
    
    This creates a single ONNX model that takes raw code strings and numeric
    features as input, performing tokenization and BERT embedding internally.
    
    Note: This requires exporting both the BERT model and fusion model separately,
    then combining them. For simplicity, we export a wrapper that handles this.
    
    Args:
        model: Trained GatedFusionModel
        tokenizer: GraphCodeBERT tokenizer
        input_dim: Dimension of numeric features
        output_path: Path to save ONNX model
        device: Device to run export on
        max_length: Maximum sequence length for BERT
        opset_version: ONNX opset version
        
    Returns:
        Dictionary with export metadata and usage instructions
    """
    from transformers import AutoModel
    from src.networks.fusion import GatedFusionModel, LateFusionMLPModel
    
    if device is None:
        device = torch.device('cpu')
    
    if not isinstance(model, (GatedFusionModel, LateFusionMLPModel)):
        raise ValueError("Model must be GatedFusionModel or LateFusionMLPModel")
    
    # Load BERT model
    bert = AutoModel.from_pretrained(GatedFusionModel.BERT_MODEL)
    bert.eval()
    bert = bert.to(device)
    model.eval()
    model = model.to(device)
    
    class FusionEndToEnd(nn.Module):
        def __init__(self, bert_model, fusion_model):
            super().__init__()
            self.bert = bert_model
            self.fusion = fusion_model
        
        def forward(self, features, input_ids, attention_mask):
            with torch.no_grad():
                bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                bert_embed = bert_out.last_hidden_state[:, 0, :]
            return self.fusion(features, bert_embed)
    
    wrapper = FusionEndToEnd(bert, model)
    wrapper.eval()
    
    # Create dummy inputs
    dummy_features = torch.randn(1, input_dim, device=device)
    dummy_input_ids = torch.randint(0, 50000, (1, max_length), device=device)
    dummy_attention_mask = torch.ones((1, max_length), device=device)
    
    # Export
    torch.onnx.export(
        wrapper,
        (dummy_features, dummy_input_ids, dummy_attention_mask),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['features', 'input_ids', 'attention_mask'],
        output_names=['output'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    return {
        'model_path': output_path,
        'input_dim': input_dim,
        'max_length': max_length,
        'opset_version': opset_version,
        'inputs': ['features', 'input_ids', 'attention_mask'],
        'note': 'This model includes BERT. Tokenization must be done before inference.'
    }


def validate_onnx_model(
    model_path: str,
    test_inputs: Dict[str, np.ndarray],
    expected_output_shape: Tuple[int, int] = None
) -> Dict[str, Any]:
    """
    Validate an exported ONNX model.
    
    Args:
        model_path: Path to ONNX model
        test_inputs: Dictionary of input name -> numpy array
        expected_output_shape: Expected shape of output (optional)
        
    Returns:
        Validation results dictionary
    """
    import time
    
    session = ort.InferenceSession(model_path)
    
    # Get model info
    input_info = [(inp.name, inp.type) for inp in session.get_inputs()]
    output_info = [(out.name, out.type) for out in session.get_outputs()]
    
    # Run inference
    start_time = time.time()
    outputs = session.run(None, test_inputs)
    inference_time = time.time() - start_time
    
    result = {
        'model_path': model_path,
        'inputs': input_info,
        'outputs': output_info,
        'output_shape': outputs[0].shape,
        'inference_time_ms': inference_time * 1000,
        'valid': True
    }
    
    if expected_output_shape:
        if len(expected_output_shape) == 2:
            # Allow variable batch size
            expected = (-1, expected_output_shape[1])
            actual = (-1, outputs[0].shape[1])
            result['shape_match'] = expected == actual
        else:
            result['shape_match'] = outputs[0].shape == expected_output_shape
    
    return result


def benchmark_onnx_vs_pytorch(
    pytorch_model: nn.Module,
    onnx_path: str,
    input_shapes: Dict[str, Tuple[int, ...]],
    device: torch.device = None,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark ONNX vs PyTorch inference speed.
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        input_shapes: Dictionary of input name -> shape
        device: Device for PyTorch inference
        num_runs: Number of runs for averaging
        
    Returns:
        Benchmark results
    """
    import time
    
    if device is None:
        device = torch.device('cpu')
    
    # Create test inputs
    torch_inputs = {}
    onnx_inputs = {}
    for name, shape in input_shapes.items():
        tensor = torch.randn(*shape, device=device)
        torch_inputs[name] = tensor
        onnx_inputs[name] = tensor.cpu().numpy()
    
    # Warmup
    with torch.no_grad():
        pytorch_model(**torch_inputs)
    
    ort_session = ort.InferenceSession(onnx_path)
    ort_session.run(None, onnx_inputs)
    
    # Benchmark PyTorch
    pytorch_model.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            pytorch_model(**torch_inputs)
    pytorch_time = (time.time() - start) / num_runs * 1000
    
    # Benchmark ONNX
    start = time.time()
    for _ in range(num_runs):
        ort_session.run(None, onnx_inputs)
    onnx_time = (time.time() - start) / num_runs * 1000
    
    return {
        'pytorch_avg_ms': pytorch_time,
        'onnx_avg_ms': onnx_time,
        'speedup': pytorch_time / onnx_time if onnx_time > 0 else float('inf'),
        'device': str(device)
    }
