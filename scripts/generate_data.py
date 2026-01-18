#!/usr/bin/env python3
"""
Generate training data for feasibility learning.

Usage:
    python scripts/generate_data.py --num_samples 10000 --output data/
    python scripts/generate_data.py --visualize --num_samples 10
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generation.collector import DataCollector


def main():
    parser = argparse.ArgumentParser(description="Generate feasibility training data")
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/",
        help="Output directory"
    )
    parser.add_argument(
        "--render", 
        action="store_true",
        help="Render PyBullet GUI"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Visualize some samples after generation"
    )
    parser.add_argument(
        "--num_visualize",
        type=int,
        default=5,
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--actions_per_scene",
        type=int,
        default=5,
        help="Number of actions to attempt per scene"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("FEASIBILITY DATA GENERATION")
    print("="*60)
    print(f"Samples to generate: {args.num_samples}")
    print(f"Output directory: {args.output}")
    print(f"Random seed: {args.seed}")
    print("="*60)
    
    # Create collector
    collector = DataCollector(
        output_dir=args.output,
        render=args.render,
        save_images=True,
        save_trajectories=True
    )
    
    try:
        # Collect data
        dataset_path = collector.collect(
            num_samples=args.num_samples,
            num_actions_per_scene=args.actions_per_scene,
            seed=args.seed
        )
        
        # Visualize samples
        if args.visualize:
            print(f"\nVisualizing {args.num_visualize} samples...")
            for i in range(args.num_visualize):
                collector.visualize_sample(i, dataset_path)
            print(f"Visualizations saved to {args.output}")
            
    finally:
        collector.close()
        
    print("\nDone!")


if __name__ == "__main__":
    main()
