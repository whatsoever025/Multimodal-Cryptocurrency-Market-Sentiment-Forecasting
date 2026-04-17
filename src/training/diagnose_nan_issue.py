"""
Comprehensive NaN Detection and Diagnosis Script

Identifies:
1. NaN/Inf values in raw embeddings and features
2. Numerical issues in forward pass
3. Which batch/sample causes NaN during backward
4. Attention score magnitudes (overflow risk)
5. Gradient magnitudes before NaN
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import create_config
from src.training.dataset import CryptoMultimodalDataset, multimodal_collate_fn
from src.training.model import MultimodalFusionNet
from torch.utils.data import DataLoader


class NaNDetector:
    """Detects and reports sources of NaN values in training pipeline."""
    
    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device
        self.model = MultimodalFusionNet(config.model).to(device)
        self.model.eval()
        
        logger.info("✓ NaN Detector initialized")
    
    @staticmethod
    def check_tensor(tensor: torch.Tensor, name: str, threshold: float = 1e6) -> Dict:
        """Check tensor for NaN, Inf, and extreme values."""
        result = {
            "name": name,
            "shape": tensor.shape,
            "dtype": tensor.dtype,
            "device": tensor.device,
            "has_nan": torch.isnan(tensor).any().item(),
            "has_inf": torch.isinf(tensor).any().item(),
            "nan_count": torch.isnan(tensor).sum().item(),
            "inf_count": torch.isinf(tensor).sum().item(),
            "min": tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].min().item() if tensor.numel() > 0 else None,
            "max": tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].max().item() if tensor.numel() > 0 else None,
            "mean": tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].mean().item() if tensor.numel() > 0 else None,
            "std": tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].std().item() if tensor.numel() > 0 else None,
            "has_extreme": (torch.abs(tensor) > threshold).any().item() if tensor.dtype in [torch.float32, torch.float64] else False,
            "extreme_count": (torch.abs(tensor) > threshold).sum().item() if tensor.dtype in [torch.float32, torch.float64] else 0,
        }
        return result
    
    @staticmethod
    def report_tensor(result: Dict) -> str:
        """Format tensor check result as readable report."""
        status = "✓" if not (result["has_nan"] or result["has_inf"]) else "✗"
        
        report = f"\n{status} {result['name']}"
        report += f"\n  Shape: {result['shape']} | DType: {result['dtype']} | Device: {result['device']}"
        
        if result["has_nan"]:
            report += f"\n  ⚠ HAS NaN ({result['nan_count']} values)"
        if result["has_inf"]:
            report += f"\n  ⚠ HAS INF ({result['inf_count']} values)"
        if result["has_extreme"]:
            report += f"\n  ⚠ EXTREME VALUES ({result['extreme_count']} > 1e6)"
        
        if result["min"] is not None:
            report += f"\n  Range: [{result['min']:.6e}, {result['max']:.6e}]"
            report += f" | Mean: {result['mean']:.6e} | Std: {result['std']:.6e}"
        
        return report
    
    def diagnose_embeddings(self, data_dir: str) -> None:
        """Check raw embeddings for NaN/Inf."""
        logger.info("\n" + "="*70)
        logger.info("STEP 1: CHECKING RAW EMBEDDINGS FOR NaN/INF")
        logger.info("="*70)
        
        features_dir = Path(data_dir) / "features"
        
        for name in ["text_embeddings.pt", "image_embeddings.pt", "tabular_features.pt", "target_scores.pt"]:
            path = features_dir / name
            if not path.exists():
                logger.warning(f"✗ Not found: {path}")
                continue
            
            try:
                data = torch.load(path)
                result = self.check_tensor(data, name)
                logger.info(self.report_tensor(result))
            except Exception as e:
                logger.error(f"✗ Failed to load {name}: {e}")
    
    def diagnose_dataset(self, dataset: CryptoMultimodalDataset, num_samples: int = 100) -> None:
        """Check dataset samples for NaN/Inf."""
        logger.info("\n" + "="*70)
        logger.info("STEP 2: CHECKING DATASET SAMPLES FOR NaN/INF")
        logger.info("="*70)
        
        problematic_samples = []
        
        for idx in tqdm(range(min(num_samples, len(dataset))), desc="Checking dataset"):
            try:
                sample = dataset[idx]
                
                has_issue = False
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        if torch.isnan(value).any() or torch.isinf(value).any():
                            has_issue = True
                            break
                
                if has_issue:
                    problematic_samples.append(idx)
            except Exception as e:
                logger.warning(f"✗ Sample {idx}: {e}")
        
        if problematic_samples:
            logger.warning(f"✗ Found NaN/Inf in {len(problematic_samples)} samples: {problematic_samples[:10]}")
        else:
            logger.info(f"✓ All {min(num_samples, len(dataset))} samples are clean (no NaN/Inf)")
    
    def diagnose_forward_pass(self, dataloader: DataLoader, num_batches: int = 10) -> None:
        """Check forward pass for NaN/Inf in intermediate tensors."""
        logger.info("\n" + "="*70)
        logger.info("STEP 3: CHECKING FORWARD PASS FOR NaN/INF")
        logger.info("="*70)
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Forward pass")):
                if batch_idx >= num_batches:
                    break
                
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                try:
                    predictions = self.model(batch)
                    
                    # Check output
                    result = self.check_tensor(predictions, f"Batch {batch_idx} predictions")
                    logger.info(self.report_tensor(result))
                    
                    if result["has_nan"] or result["has_inf"]:
                        logger.warning(f"✗ Batch {batch_idx} produces NaN/Inf!")
                        logger.info(f"  Batch keys: {batch.keys()}")
                        logger.info(f"  Batch shapes: {[(k, v.shape if isinstance(v, torch.Tensor) else 'non-tensor') for k, v in batch.items()]}")
                        
                        # Check input shapes
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                result_in = self.check_tensor(value, f"  {key}")
                                logger.info(self.report_tensor(result_in))
                
                except Exception as e:
                    logger.error(f"✗ Batch {batch_idx} forward failed: {e}")
                    import traceback
                    traceback.print_exc()
    
    def diagnose_backward_pass(self, dataloader: DataLoader, num_batches: int = 5) -> None:
        """Check backward pass for NaN in gradients."""
        logger.info("\n" + "="*70)
        logger.info("STEP 4: CHECKING BACKWARD PASS FOR NaN IN GRADIENTS")
        logger.info("="*70)
        
        torch.autograd.set_detect_anomaly(True)
        
        # Create optimizer (required for backward)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        self.model.train()
        
        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc="Backward pass")):
            if batch_idx >= num_batches:
                break
            
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            try:
                # Forward
                predictions = self.model(batch)
                targets = batch["target"]
                
                # Loss
                loss = nn.HuberLoss(delta=1.0)(predictions, targets)
                
                # Check loss for NaN
                loss_result = self.check_tensor(loss.unsqueeze(0), f"Batch {batch_idx} loss")
                logger.info(self.report_tensor(loss_result))
                
                if loss_result["has_nan"]:
                    logger.error(f"✗ Batch {batch_idx} loss is NaN before backward!")
                    continue
                
                # Backward
                loss.backward()
                
                # Check gradients for NaN
                logger.info(f"\n  Checking gradients for Batch {batch_idx}:")
                has_grad_nan = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_result = self.check_tensor(param.grad, f"  {name}.grad", threshold=1e4)
                        if grad_result["has_nan"] or grad_result["has_inf"] or grad_result["has_extreme"]:
                            logger.warning(self.report_tensor(grad_result))
                            has_grad_nan = True
                        elif grad_result["extreme_count"] > 0:
                            logger.warning(f"  ⚠ {name}: {grad_result['extreme_count']} extreme gradient values")
                
                if not has_grad_nan:
                    logger.info(f"  ✓ All gradients are clean")
                
                optimizer.zero_grad()
            
            except RuntimeError as e:
                if "nan" in str(e).lower():
                    logger.error(f"✗ Batch {batch_idx}: {e}")
                    logger.info(f"  This is the problematic batch! Saving details...")
                    
                    # Save batch for inspection
                    batch_path = Path("/tmp/problematic_batch.pt")
                    torch.save({
                        "batch_idx": batch_idx,
                        "batch": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
                        "error": str(e)
                    }, batch_path)
                    logger.info(f"  Saved to: {batch_path}")
                    break
                else:
                    raise


def main(args):
    """Run NaN diagnosis."""
    # Create config
    config = create_config(args.config)
    
    logger.info("\n" + "="*70)
    logger.info("NaN DIAGNOSIS SUITE")
    logger.info("="*70)
    logger.info(f"Config: {args.config}")
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Device: {args.device}")
    
    # Initialize detector
    detector = NaNDetector(config, device=args.device)
    
    # Step 1: Check raw embeddings
    detector.diagnose_embeddings(args.data_dir)
    
    # Step 2: Check dataset samples
    try:
        dataset = CryptoMultimodalDataset(
            split="train",
            seq_len=config.data.seq_len,
            features_dir=str(Path(args.data_dir) / "features")
        )
        detector.diagnose_dataset(dataset, num_samples=args.num_samples)
        
        # Step 3: Check forward pass
        dataloader = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            collate_fn=multimodal_collate_fn,
            num_workers=0
        )
        detector.diagnose_forward_pass(dataloader, num_batches=args.num_batches)
        
        # Step 4: Check backward pass
        detector.diagnose_backward_pass(dataloader, num_batches=args.num_batches // 2)
    
    except Exception as e:
        logger.error(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSIS COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose NaN issues in training pipeline")
    parser.add_argument("--config", type=str, default="v0", help="Config name")
    parser.add_argument("--data-dir", type=str, default="/kaggle/input/crypto-dataset", help="Data directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of dataset samples to check")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches to check")
    
    args = parser.parse_args()
    main(args)
