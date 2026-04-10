#!/bin/bash
###############################################################################
# Training Script for Kaggle/Local Execution
# 
# Usage: 
#   bash train.sh --asset BTC --skip-crawlers --skip-data-prep
#   bash train.sh --asset BTC --resume  # Resume from latest checkpoint
# 
# This script orchestrates:
# 1. Loading environment variables from .env
# 2. Installing dependencies
# 3. Data preparation (optional)
# 4. Model training (with optional resume capability)
###############################################################################

set -e  # Exit on error

# Default values
ASSET="BTC"
SKIP_CRAWLERS=False
SKIP_DATA_PREP=False
RESUME=False

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --asset)
            ASSET="$2"
            shift 2
            ;;
        --skip-crawlers)
            SKIP_CRAWLERS=True
            shift
            ;;
        --skip-data-prep)
            SKIP_DATA_PREP=True
            shift
            ;;
        --resume)
            RESUME=True
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo "TRAINING SCRIPT"
echo "================================================================================"
echo "Asset: $ASSET"
echo "Skip Crawlers: $SKIP_CRAWLERS"
echo "Skip Data Prep: $SKIP_DATA_PREP"
echo "Resume from Checkpoint: $RESUME"
echo ""

# Step 1: Load environment variables
echo "Step 1: Loading environment variables..."
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "✓ Environment variables loaded from .env"
else
    echo "⚠ .env file not found"
fi

# Step 2: Install requirements
echo ""
echo "Step 2: Installing requirements..."
python -m pip install -q -r requirements.txt
echo "✓ Requirements installed"

# Step 3: Data preparation (optional)
if [ "$SKIP_DATA_PREP" != "True" ]; then
    echo ""
    echo "Step 3: Data preparation..."
    
    if [ "$SKIP_CRAWLERS" != "True" ]; then
        echo "  3A: Downloading datasets from HuggingFace..."
        python -c "
import sys
sys.path.insert(0, 'src')
from huggingface_loader import download_hf_dataset
download_hf_dataset(asset='$ASSET', output_dir='data/raw')
" || echo "  ⚠ HuggingFace download completed with warnings"
    else
        echo "  3A: Skipping HuggingFace download"
    fi
    
    echo "  3B: Aligning multimodal data..."
    python -c "
import sys
sys.path.insert(0, 'src')
from preprocessing.data_aligner import DataAligner
aligner = DataAligner(asset='$ASSET')
aligner.align_all()
" || echo "  ⚠ Data alignment completed with warnings"
    
    echo "  3C: Generating candlestick charts..."
    python -c "
import sys
sys.path.insert(0, 'src')
from preprocessing.chart_generator import ChartGenerator
generator = ChartGenerator()
generator.generate_charts_from_csv('data/raw/${ASSET}USDT_klines.csv', symbol='${ASSET,,}', num_workers=4)
" || echo "  ⚠ Chart generation completed with warnings"
    
    echo "✓ Data preparation complete"
else
    echo "Step 3: Skipping data preparation"
fi

# Step 4: Train model
echo ""
echo "Step 4: Training model..."
echo "================================================================================"
TRAIN_CMD="python src/training/train.py --asset $ASSET"
if [ "$RESUME" = "True" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume"
fi
$TRAIN_CMD

echo ""
echo "================================================================================"
echo "✓ TRAINING COMPLETE"
echo "================================================================================"
