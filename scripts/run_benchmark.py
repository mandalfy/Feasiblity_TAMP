#!/usr/bin/env python3
"""
Benchmark baseline vs ML-guided TAMP planners.

Usage:
    python scripts/run_benchmark.py --num_scenarios 50
    python scripts/run_benchmark.py --num_scenarios 10 --quick
"""

import argparse
import sys
import os
import time
import json
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt

from environments.tabletop_env import TabletopEnv
from planning.baseline_tamp import BaselineTAMPPlanner
from planning.ml_guided_tamp import MLGuidedTAMPPlanner
from models.mlp_classifier import MLPClassifier
from data_generation.dataset import FeasibilityDataset


def load_model(checkpoint_path: str, data_path: str) -> MLPClassifier:
    """Load trained model from checkpoint."""
    dataset = FeasibilityDataset(data_path)
    _, _, input_dim = dataset.get_input_dims()
    
    model = MLPClassifier(input_dim=input_dim)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def create_random_goal(env: TabletopEnv) -> Dict[int, np.ndarray]:
    """Create a random goal for one object."""
    if not env.object_ids:
        return {}
        
    # Pick random object
    obj_id = np.random.choice(env.object_ids)
    
    # Random target position on table
    target = env.sample_place_target()
    
    return {obj_id: target}


def run_benchmark(
    num_scenarios: int,
    checkpoint_path: str,
    data_path: str,
    timeout: float = 30.0,
    quick: bool = False
) -> Dict:
    """
    Run benchmark comparing baseline and ML-guided planners.
    
    Returns:
        Dict with benchmark results
    """
    results = {
        'baseline': {
            'times': [],
            'successes': [],
            'motion_planning_calls': [],
            'nodes_expanded': []
        },
        'ml_guided': {
            'times': [],
            'successes': [],
            'motion_planning_calls': [],
            'nodes_expanded': [],
            'pruned_actions': []
        }
    }
    
    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path, data_path)
    
    # Reduce timeout for quick mode
    if quick:
        timeout = 10.0
        
    print(f"\nRunning {num_scenarios} scenarios...")
    print("-" * 60)
    
    for scenario_idx in range(num_scenarios):
        print(f"\nScenario {scenario_idx + 1}/{num_scenarios}")
        
        # Create fresh environment for each scenario
        env = TabletopEnv(render=False, seed=scenario_idx)
        env.reset(num_objects=3)
        
        # Create goal
        goal = create_random_goal(env)
        if not goal:
            continue
            
        # Run baseline planner
        print("  Baseline planner...", end=" ")
        baseline_planner = BaselineTAMPPlanner(env)
        baseline_result = baseline_planner.plan(goal, timeout=timeout)
        
        results['baseline']['times'].append(baseline_result.planning_time)
        results['baseline']['successes'].append(baseline_result.success)
        results['baseline']['motion_planning_calls'].append(baseline_result.motion_planning_calls)
        results['baseline']['nodes_expanded'].append(baseline_result.nodes_expanded)
        
        status = "✓" if baseline_result.success else "✗"
        print(f"{status} ({baseline_result.planning_time:.2f}s, {baseline_result.motion_planning_calls} MP calls)")
        
        # Reset environment to same state
        env.reset(num_objects=3)
        np.random.seed(scenario_idx)  # Same seed for same goal
        goal = create_random_goal(env)
        
        # Run ML-guided planner
        print("  ML-guided planner...", end=" ")
        ml_planner = MLGuidedTAMPPlanner(
            env, 
            model,
            feasibility_threshold=0.3
        )
        ml_result = ml_planner.plan(goal, timeout=timeout)
        stats = ml_planner.get_statistics()
        
        results['ml_guided']['times'].append(ml_result.planning_time)
        results['ml_guided']['successes'].append(ml_result.success)
        results['ml_guided']['motion_planning_calls'].append(ml_result.motion_planning_calls)
        results['ml_guided']['nodes_expanded'].append(ml_result.nodes_expanded)
        results['ml_guided']['pruned_actions'].append(stats['pruned_actions'])
        
        status = "✓" if ml_result.success else "✗"
        print(f"{status} ({ml_result.planning_time:.2f}s, {ml_result.motion_planning_calls} MP calls, {stats['pruned_actions']} pruned)")
        
        env.close()
        
    return results


def analyze_results(results: Dict) -> Dict:
    """Analyze benchmark results."""
    analysis = {}
    
    for planner in ['baseline', 'ml_guided']:
        data = results[planner]
        
        times = np.array(data['times'])
        successes = np.array(data['successes'])
        mp_calls = np.array(data['motion_planning_calls'])
        nodes = np.array(data['nodes_expanded'])
        
        analysis[planner] = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'median_time': np.median(times),
            'success_rate': np.mean(successes),
            'avg_mp_calls': np.mean(mp_calls),
            'avg_nodes': np.mean(nodes),
        }
        
    # Compute speedup
    if analysis['baseline']['avg_time'] > 0:
        analysis['speedup'] = analysis['baseline']['avg_time'] / max(0.001, analysis['ml_guided']['avg_time'])
    else:
        analysis['speedup'] = 1.0
        
    # MP call reduction
    if analysis['baseline']['avg_mp_calls'] > 0:
        analysis['mp_reduction'] = 1 - (analysis['ml_guided']['avg_mp_calls'] / analysis['baseline']['avg_mp_calls'])
    else:
        analysis['mp_reduction'] = 0.0
        
    return analysis


def print_results(analysis: Dict):
    """Print benchmark results."""
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    print("\n{:<25} {:>15} {:>15}".format("Metric", "Baseline", "ML-Guided"))
    print("-"*60)
    
    print("{:<25} {:>15.3f}s {:>15.3f}s".format(
        "Avg Planning Time",
        analysis['baseline']['avg_time'],
        analysis['ml_guided']['avg_time']
    ))
    
    print("{:<25} {:>15.1f}% {:>15.1f}%".format(
        "Success Rate",
        analysis['baseline']['success_rate'] * 100,
        analysis['ml_guided']['success_rate'] * 100
    ))
    
    print("{:<25} {:>15.1f} {:>15.1f}".format(
        "Avg MP Calls",
        analysis['baseline']['avg_mp_calls'],
        analysis['ml_guided']['avg_mp_calls']
    ))
    
    print("{:<25} {:>15.1f} {:>15.1f}".format(
        "Avg Nodes Expanded",
        analysis['baseline']['avg_nodes'],
        analysis['ml_guided']['avg_nodes']
    ))
    
    print("-"*60)
    print(f"\n🚀 SPEEDUP: {analysis['speedup']:.2f}x")
    print(f"📉 MP Call Reduction: {analysis['mp_reduction']*100:.1f}%")
    print("="*60)


def plot_results(results: Dict, analysis: Dict, save_path: str = "plots/benchmark.png"):
    """Generate benchmark plots."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Planning time comparison
    ax = axes[0, 0]
    ax.boxplot(
        [results['baseline']['times'], results['ml_guided']['times']],
        labels=['Baseline', 'ML-Guided']
    )
    ax.set_ylabel('Planning Time (s)')
    ax.set_title('Planning Time Distribution')
    ax.grid(True, alpha=0.3)
    
    # Motion planning calls
    ax = axes[0, 1]
    x = range(len(results['baseline']['motion_planning_calls']))
    ax.plot(x, results['baseline']['motion_planning_calls'], 'b-', alpha=0.7, label='Baseline')
    ax.plot(x, results['ml_guided']['motion_planning_calls'], 'g-', alpha=0.7, label='ML-Guided')
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Motion Planning Calls')
    ax.set_title('Motion Planning Calls per Scenario')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary bar chart
    ax = axes[1, 0]
    metrics = ['Avg Time (s)', 'MP Calls', 'Nodes']
    baseline_vals = [
        analysis['baseline']['avg_time'],
        analysis['baseline']['avg_mp_calls'] / 10,  # Scale for visibility
        analysis['baseline']['avg_nodes'] / 10
    ]
    ml_vals = [
        analysis['ml_guided']['avg_time'],
        analysis['ml_guided']['avg_mp_calls'] / 10,
        analysis['ml_guided']['avg_nodes'] / 10
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#3498db')
    ax.bar(x + width/2, ml_vals, width, label='ML-Guided', color='#2ecc71')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title('Performance Comparison')
    ax.legend()
    
    # Speedup summary
    ax = axes[1, 1]
    ax.text(0.5, 0.6, f"Speedup: {analysis['speedup']:.2f}x", 
            ha='center', va='center', fontsize=32, fontweight='bold', color='#27ae60')
    ax.text(0.5, 0.3, f"MP Reduction: {analysis['mp_reduction']*100:.1f}%",
            ha='center', va='center', fontsize=24, color='#2980b9')
    ax.axis('off')
    ax.set_title('Summary')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TAMP planners")
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=50,
        help="Number of scenarios to test"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/feasibility_dataset.h5",
        help="Path to dataset (for model dimensions)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Planning timeout per scenario"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with reduced timeout"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.checkpoint):
        print(f"Error: Model checkpoint not found at {args.checkpoint}")
        print("Run 'python scripts/train_model.py' first.")
        sys.exit(1)
        
    if not os.path.exists(args.data):
        print(f"Error: Dataset not found at {args.data}")
        print("Run 'python scripts/generate_data.py' first.")
        sys.exit(1)
        
    print("="*60)
    print("TAMP PLANNER BENCHMARK")
    print("="*60)
    print(f"Scenarios: {args.num_scenarios}")
    print(f"Timeout: {args.timeout}s")
    print(f"Model: {args.checkpoint}")
    print("="*60)
    
    # Run benchmark
    results = run_benchmark(
        args.num_scenarios,
        args.checkpoint,
        args.data,
        args.timeout,
        args.quick
    )
    
    # Analyze and display
    analysis = analyze_results(results)
    print_results(analysis)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump({
            'results': {k: {kk: list(vv) if isinstance(vv, np.ndarray) else vv 
                           for kk, vv in v.items()} 
                       for k, v in results.items()},
            'analysis': analysis
        }, f, indent=2, default=float)
    print(f"\nResults saved to {args.output}")
    
    # Plot
    plot_results(results, analysis)


if __name__ == "__main__":
    main()
