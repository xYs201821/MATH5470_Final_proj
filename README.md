# Financial Time Series Prediction using CNN

This project implements a Convolutional Neural Network (CNN) for predicting financial time series returns. The model processes market data images to predict returns over different time horizons (5, 20, and 60 days).

## Project Structure

```
.
├── config.yaml           # Model configuration parameters
├── dataset.py           # Dataset loading and preprocessing
├── eval.py             # Model evaluation and metrics calculation
├── experiment.py       # Hyperparameter sensitivity analysis
├── grad-cam.py         # Gradient-weighted Class Activation Mapping visualization
├── model.py            # CNN model architecture
├── train.py            # Model training script
├── train_utils.py      # Training utilities and metrics
├── utils.py            # Helper functions
├── model/              # Trained model checkpoints
│   ├── baseline_I20R5.pth
│   ├── baseline_I20R20.pth
│   ├── baseline_I20R60.pth
│   └── regression.pth
└── output/             # Evaluation results and visualizations
    ├── 5d/            # 5-day prediction results
    ├── 20d/           # 20-day prediction results
    ├── 60d/           # 60-day prediction results
    ├── sensitivity/   # Sensitivity analysis results
    └── grad-cam/      # Grad-CAM visualizations
```

## Features

- CNN-based model for financial time series prediction
- Support for multiple prediction horizons (5, 20, 60 days)
- Hyperparameter sensitivity analysis
- Model interpretability using Grad-CAM
- Comprehensive evaluation metrics including:
  - Accuracy, Precision, Recall, F1 Score
  - Value-weighted and equal-weighted Sharpe ratios

## Requirements

- Python 3.12
- PyTorch
- NumPy
- Pandas
- OpenCV (for Grad-CAM visualization)
- PyYAML
- tqdm

## Usage

### Training

To train the baseline model:

```bash
python train.py -model_path ./model -data ./monthly_20d -output ./output -config_path ./config.yaml -model_name baseline_I20R20 -ret_days 20
```

### Evaluation

To evaluate the model on different time horizons:

```bash
# Evaluate 5-day predictions
python eval.py -model_path ./model -model_name baseline_I20R20 -device cuda -config_path ./config.yaml -data monthly_20d -ret_days 5 -output output/5d

# Evaluate 20-day predictions
python eval.py -model_path ./model -model_name baseline_I20R20 -device cuda -config_path ./config.yaml -data monthly_20d -ret_days 20 -output output/20d

# Evaluate 60-day predictions
python eval.py -model_path ./model -model_name baseline_I20R20 -device cuda -config_path ./config.yaml -data monthly_20d -ret_days 60 -output output/60d
```

### Hyperparameter Sensitivity Analysis

To run sensitivity analysis:

```bash
# Train sensitivity models
python experiment.py --train --output_dir output/sensitivity --model_dir model/sensitivity --ret_days 20

# Evaluate sensitivity models
python experiment.py --eval --output_dir output/sensitivity --model_dir model/sensitivity --ret_days 20
```

### Model Visualization

To generate Grad-CAM visualizations:

```bash
python grad-cam.py
```

## Model Architecture

The CNN model (`CNN20`) features:
- Configurable number of convolutional layers
- Batch normalization
- Dropout for regularization
- Multiple activation function options (ReLU, LeakyReLU, tanh, etc.)
- Xavier initialization
- Customizable kernel sizes, strides, and dilation rates

## Configuration

Model parameters can be configured in `config.yaml`:

```yaml
drop_out: 0.5
conv_filters: 64
layers: 3
batch_normalization: true
xavier_initialization: true
kernel_sizes: [5, 3]
strides: [3, 1]
dilation: [2, 1]
activation: "LReLu"
```

## Results

All the metrics are saved in the `output` directory.

Training and Validation Loss are stored in `*_metrics.csv`:
- `baseline_I20R5_metrics.csv`
- `baseline_I20R20_metrics.csv`
- `baseline_I20R60_metrics.csv`
- `regression_metrics.csv`

Model performance metrics and portfolio returns are saved in the `*.yaml` in the task's folder. 
For example, `5d/baseline_I20R20.yaml` contains the prediction performance of `baseline_I20R20` model over `5-day` period. 