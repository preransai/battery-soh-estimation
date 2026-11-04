"""
Phase-1 Complete Pipeline
Execute the entire Phase-1 workflow: Load → Segment → Extract → Train → Evaluate

Usage:
    python main_pipeline.py
"""

import sys
import os
import time
sys.path.append('src')

from utils import (
    load_config, 
    setup_logging, 
    create_directories, 
    set_random_seed,
    print_section_header,
    print_subsection_header,
    format_time
)
from data_loader import load_battery_data
from cycle_segmentation import segment_and_filter_cycles
from feature_extraction import extract_and_save_features
from baseline_model import BaselineTrainer
import pandas as pd


def main():
    """Execute Phase-1 pipeline."""
    
    # Start timer
    pipeline_start = time.time()
    
    print_section_header("PHASE-1 PIPELINE: NASA B0005 BATTERY SOH PREDICTION", "=")
    
    # Load configuration
    config = load_config('config.yaml')
    logger = setup_logging(config)
    set_random_seed(config['random_seed'])
    
    # Create output directories
    create_directories(config)
    
    # ========================================================================
    # STEP 1: DATA LOADING
    # ========================================================================
    print_section_header("STEP 1: DATA LOADING")
    step_start = time.time()
    
    loader = load_battery_data(config)
    
    step_time = time.time() - step_start
    print(f"\n✓ Step 1 completed in {format_time(step_time)}")
    
    # ========================================================================
    # STEP 2: CYCLE SEGMENTATION & FILTERING
    # ========================================================================
    print_section_header("STEP 2: CYCLE SEGMENTATION & FILTERING")
    step_start = time.time()
    
    output_dir = config['paths']['processed_data']
    filtered_cycles = segment_and_filter_cycles(loader, config, save_dir=output_dir)
    
    step_time = time.time() - step_start
    print(f"\n✓ Step 2 completed in {format_time(step_time)}")
    
    # ========================================================================
    # STEP 3: FEATURE EXTRACTION
    # ========================================================================
    print_section_header("STEP 3: PHYSICS-GUIDED FEATURE EXTRACTION")
    step_start = time.time()
    
    features_path = os.path.join(config['paths']['features'], 'features_B0005.csv')
    features_df = extract_and_save_features(filtered_cycles, config, features_path)
    
    step_time = time.time() - step_start
    print(f"\n✓ Step 3 completed in {format_time(step_time)}")
    
    # ========================================================================
    # STEP 4: BASELINE MODEL TRAINING
    # ========================================================================
    print_section_header("STEP 4: BASELINE CNN-LSTM MODEL TRAINING")
    step_start = time.time()
    
    # Initialize trainer
    trainer = BaselineTrainer(config)
    
    # Prepare data
    print_subsection_header("Preparing Data")
    train_loader, test_loader = trainer.prepare_data(features_df)
    
    # Build model
    print_subsection_header("Building Model")
    input_size = len(trainer.feature_cols)
    trainer.build_model(input_size)
    
    # Train
    print_subsection_header("Training")
    trainer.train(train_loader, test_loader)
    
    step_time = time.time() - step_start
    print(f"\n✓ Step 4 completed in {format_time(step_time)}")
    
    # ========================================================================
    # STEP 5: EVALUATION
    # ========================================================================
    print_section_header("STEP 5: MODEL EVALUATION")
    step_start = time.time()
    
    metrics, predictions, actuals = trainer.evaluate(test_loader)
    
    print(f"\n{'=' * 80}")
    print("BASELINE MODEL PERFORMANCE (NUMERIC ONLY)".center(80))
    print('=' * 80)
    print(f"\n  MAE:   {metrics['MAE']:.4f}%")
    print(f"  RMSE:  {metrics['RMSE']:.4f}%")
    print(f"  MAPE:  {metrics['MAPE']:.2f}%")
    print('\n' + '=' * 80)
    
    step_time = time.time() - step_start
    print(f"\n✓ Step 5 completed in {format_time(step_time)}")
    
    # ========================================================================
    # STEP 6: SAVE MODEL
    # ========================================================================
    print_section_header("STEP 6: SAVING MODEL")
    step_start = time.time()
    
    model_dir = config['paths']['models']
    trainer.save_model(model_dir, model_name='baseline_cnn_lstm_B0005')
    
    step_time = time.time() - step_start
    print(f"\n✓ Step 6 completed in {format_time(step_time)}")
    
    # ========================================================================
    # PIPELINE SUMMARY
    # ========================================================================
    pipeline_time = time.time() - pipeline_start
    
    print_section_header("PHASE-1 PIPELINE COMPLETED", "=")
    
    print(f"Total execution time: {format_time(pipeline_time)}")
    print(f"\nData processed:")
    print(f"  ✓ Battery: {config['dataset']['battery_id']}")
    print(f"  ✓ Cycles processed: {len(filtered_cycles)}")
    print(f"  ✓ Features extracted: {features_df.shape[1]}")
    print(f"  ✓ Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    print(f"\nOutputs saved:")
    print(f"  ✓ Processed cycles: {config['paths']['processed_data']}")
    print(f"  ✓ Features: {features_path}")
    print(f"  ✓ Model: {model_dir}")
    
    print(f"\nBaseline Performance:")
    print(f"  ✓ MAE: {metrics['MAE']:.4f}%")
    print(f"  ✓ RMSE: {metrics['RMSE']:.4f}%")
    print(f"  ✓ MAPE: {metrics['MAPE']:.2f}%")
    
    print(f"\n{'=' * 80}")
    print("NEXT STEPS (PHASE-2)".center(80))
    print('=' * 80)
    print("""
1. Feature-to-Text Conversion
   - Create text descriptions from physics features
   - Example: "ICA peak shifted from 3.7V to 3.55V"

2. LLM Embeddings
   - Use sentence transformers for semantic embeddings
   - Compare: BERT, MPNet, etc.

3. Multimodal Fusion
   - Concatenate numeric + semantic embeddings
   - Attention-based fusion

4. Explainability
   - SHAP for feature importance
   - LLM-generated explanations
   - Attention visualization

5. Validation
   - Test on B0006, B0007, B0018
   - Cross-battery generalization
""")
    
    print('=' * 80)
    print(f"\n🎉 Phase-1 COMPLETE! Baseline established.\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Pipeline failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
