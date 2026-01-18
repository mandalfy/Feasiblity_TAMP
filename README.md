# Learning Feasibility Heuristics for TAMP

A machine learning approach to accelerate Task and Motion Planning by predicting motion plan feasibility.

## Overview

Classical TAMP is computationally expensive because it must constantly check if high-level actions are geometrically feasible. This project trains a neural network to predict feasibility, enabling faster planning through intelligent search pruning.



## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate training data
python scripts/generate_data.py --num_samples 10000

# Train the feasibility classifier
python scripts/train_model.py --model mlp --epochs 50

# Run benchmark comparison
python scripts/run_benchmark.py --num_scenarios 50
```

## Project Structure

```
├── configs/           # Configuration files
├── environments/      # PyBullet simulation environment
├── data_generation/   # Motion planning & data collection
├── models/            # Neural network architectures
├── training/          # Training & evaluation code
├── planning/          # TAMP planners (baseline & ML-guided)
├── scripts/           # Entry point scripts
└── notebooks/         # Visualization & analysis
```

## Key Components

1. **Data Generation**: Collect successful/failed motion plans in PyBullet
2. **ML Classifier**: Predict feasibility from state vectors or images
3. **ML-Guided TAMP**: Prune infeasible actions before expensive planning

## License

MIT
