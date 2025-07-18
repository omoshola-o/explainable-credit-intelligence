#!/usr/bin/env python3
"""
Verify the completeness of the reviewers package
"""

import os
from pathlib import Path

def check_files():
    """Check if all required files are present"""
    
    base_path = Path(__file__).parent
    
    required_files = {
        'Paper': [
            'paper/explainable_credit_intelligence.tex'
        ],
        'Figures': [
            'figures/figure_1_wavelet_decomposition.png',
            'figures/figure_2_lstm_embeddings.png',
            'figures/figure_3_crossshap_interactions.png',
            'figures/figure_4_model_performance.png',
            'figures/figure_5_data_validation.png',
            'figures/figure_6_synthetic_data_validation.png',
            'figures/figure_7_risk_assessment_comparison.png',
            'figures/figure_8_computational_performance.png'
        ],
        'Code': [
            'code/crossshap_algorithm.py',
            'code/statistical_validation.py',
            'code/robustness_safe_ml.py'
        ],
        'Documentation': [
            'README.md',
            'documentation/supplementary_materials.md'
        ]
    }
    
    print("Verifying Reviewers Package Completeness")
    print("=" * 50)
    
    all_present = True
    
    for category, files in required_files.items():
        print(f"\n{category}:")
        for file in files:
            file_path = base_path / file
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"  ✓ {file} ({size:,} bytes)")
            else:
                print(f"  ✗ {file} (MISSING)")
                all_present = False
    
    print("\n" + "=" * 50)
    
    if all_present:
        print("✓ All files present. Package is complete!")
    else:
        print("✗ Some files are missing. Please check the package.")
    
    # Count total size
    total_size = 0
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if not file.startswith('.'):
                total_size += os.path.getsize(os.path.join(root, file))
    
    print(f"\nTotal package size: {total_size/1024/1024:.1f} MB")

if __name__ == "__main__":
    check_files()