"""
Data Explorer: Visualize and inspect the aligned multimodal dataset
"""
from src.preprocessing.data_aligner import DataAligner
import pandas as pd
import numpy as np


def print_section(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)


def explore_dataset():
    """Demonstrate the complete aligned dataset"""
    
    print_section("DataAligner: Comprehensive Data Exploration")
    
    # Initialize
    aligner = DataAligner(asset="BTC", horizon_hours=24)
    
    # Run complete pipeline
    print("\nRunning complete pipeline...")
    df = aligner.run(push_to_hub=False)
    
    # Dataset Overview
    print_section("1. DATASET OVERVIEW")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Assets: {df['asset'].unique()}")
    
    # Column Summary
    print_section("2. COLUMNS & DATA TYPES")
    print(df.dtypes)
    print(f"\nMissing values:")
    missing = df.isna().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  None! (100% complete)")
    
    # Market Data Stats
    print_section("3. MARKET DATA STATISTICS")
    print(df[['open', 'high', 'low', 'close', 'volume']].describe())
    
    # Sentiment Target
    print_section("4. REGRESSION TARGET (sentiment_score)")
    target = df['sentiment_score']
    print(f"Min: {target.min():.2f}")
    print(f"Max: {target.max():.2f}")
    print(f"Mean: {target.mean():.2f}")
    print(f"Median: {target.median():.2f}")
    print(f"Std Dev: {target.std():.2f}")
    print(f"Skewness: {target.skew():.2f}")
    print(f"Kurtosis: {target.kurtosis():.2f}")
    
    # Funding Rates
    print_section("5. PERPETUAL FUNDING RATES")
    fr = df['funding_rate'].dropna()
    print(f"Available: {len(fr):,} records ({100*len(fr)/len(df):.1f}%)")
    print(f"Min: {fr.min():.6f}")
    print(f"Max: {fr.max():.6f}")
    print(f"Mean: {fr.mean():.6f}")
    
    # On-Chain Metrics
    print_section("6. ON-CHAIN METRICS")
    print(f"Open Interest:")
    oi = df['open_interest_close'].dropna()
    print(f"  Available: {len(oi):,} records ({100*len(oi)/len(df):.1f}%)")
    print(f"  Range: ${oi.min():.0f} to ${oi.max():.0f}")
    print(f"  Mean: ${oi.mean():.0f}")
    
    print(f"\nLiquidations (Long):")
    long_liq = df['long_liquidations']
    print(f"  Non-zero: {(long_liq > 0).sum():,} hours")
    print(f"  Mean: {long_liq.mean():.2f} BTC")
    
    print(f"\nLiquidations (Short):")
    short_liq = df['short_liquidations']
    print(f"  Non-zero: {(short_liq > 0).sum():,} hours")
    print(f"  Mean: {short_liq.mean():.2f} BTC")
    
    print(f"\nLong/Short Ratio:")
    lsr = df['long_short_ratio'].dropna()
    print(f"  Available: {len(lsr):,} records ({100*len(lsr)/len(df):.1f}%)")
    print(f"  Range: {lsr.min():.2f} to {lsr.max():.2f}")
    print(f"  Mean: {lsr.mean():.2f} (1.0 = neutral, >1.0 = bullish)")
    
    # Sentiment Indices
    print_section("7. SENTIMENT INDICES")
    print(f"Fear & Greed Index:")
    fg = df['fear_greed_value'].dropna()
    print(f"  Available: {len(fg):,} records ({100*len(fg)/len(df):.1f}%)")
    print(f"  Range: {fg.min():.0f} to {fg.max():.0f}")
    print(f"  Classifications: {df['fear_greed_classification'].unique()}")
    
    print(f"\nMacroeconomic News Volume:")
    nv = df['news_volume']
    print(f"  Total articles: {nv.sum():,}")
    print(f"  Max per hour: {nv.max()}")
    print(f"  Mean per hour: {nv.mean():.1f}")
    
    print(f"\nMacro Sentiment Tone:")
    tone = df['avg_sentiment_tone']
    print(f"  Min (negative): {tone.min():.2f}")
    print(f"  Max (positive): {tone.max():.2f}")
    print(f"  Mean: {tone.mean():.2f}")
    
    # Text Data
    print_section("8. TEXT DATA (AGGREGATED HOURLY)")
    text = df['text_content']
    placeholder_count = (text == '[NO_EVENT] market is quiet').sum()
    print(f"Total hours: {len(text):,}")
    print(f"Hours with text: {(text != '[NO_EVENT] market is quiet').sum():,} ({100*(1-placeholder_count/len(text)):.1f}%)")
    print(f"Hours with placeholder: {placeholder_count:,} ({100*placeholder_count/len(text):.1f}%)")
    
    print(f"\nSample text snippets:")
    for i in range(3):
        sample = text.iloc[i*5000]
        if sample != '[NO_EVENT] market is quiet':
            print(f"  [{df['timestamp'].iloc[i*5000]}] {sample[:100]}...")
    
    # Images
    print_section("9. CHART IMAGES")
    images = df['image_path']
    print(f"Total images: {len(images):,}")
    print(f"Sample paths:")
    for i in range(3):
        print(f"  {images.iloc[i*10000]}")
    
    # Correlations
    print_section("10. FEATURE CORRELATIONS WITH TARGET")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    correlations = df[numeric_cols].corrwith(df['sentiment_score']).sort_values(ascending=False)
    print(correlations.head(10))
    
    # Time Series Coverage
    print_section("11. TIME SERIES COVERAGE")
    print(f"Total span: {(df['timestamp'].max() - df['timestamp'].min()).days} days ({(df['timestamp'].max() - df['timestamp'].min()).days // 365} years)")
    print(f"Hourly frequency: {len(df):,} hours")
    print(f"Expected (continuous): {(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600:.0f} hours")
    
    # Gap analysis
    timestamps = df['timestamp'].sort_values()
    gaps = timestamps.diff()
    max_gap = gaps.max()
    if pd.Timedelta(hours=1) in gaps.values:
        print(f"Continuity: High (mostly 1-hour intervals)")
    else:
        print(f"Max gap: {max_gap}")
    
    # Sample records
    print_section("12. SAMPLE RECORDS")
    print(f"\nFirst record (oldest):")
    print(df.iloc[0][['timestamp', 'asset', 'close', 'sentiment_score', 'fear_greed_value', 'text_content']])
    
    print(f"\nLastmost record (newest):")
    print(df.iloc[-1][['timestamp', 'asset', 'close', 'sentiment_score', 'fear_greed_value', 'text_content']])
    
    print_section("✓ EXPLORATION COMPLETE")
    print(f"\nDataset successfully created and ready for modeling!")
    print(f"\nNext steps:")
    print(f"1. Save dataset: df.to_parquet('aligned_sentiment_btc_24h.parquet')")
    print(f"2. Train model: model.fit(X=df.drop('sentiment_score', axis=1), y=df['sentiment_score'])")
    print(f"3. Push to Hub: aligner.run(push_to_hub=True)")


if __name__ == "__main__":
    explore_dataset()
