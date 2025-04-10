
import yaml
import os
import subprocess
import argparse
import sys
experiment_variations = {
    "drop_out": [0, 0.25, 0.75],
    "conv_filters": [32,  128],
    "layers": [2, 4],
    "batch_normalization": [False],
    "xavier_initialization": [False], 
    "kernel_sizes": [[7, 3], [3, 3]],
    "strides": [[1, 1]],
    "dilation": [[1, 1]],
    "activation": ["ReLu", "tanh"],
}

def train_exp(experiments, hyperparams, ret_days=5,output_dir='./model/experiment'):
    config = yaml.safe_load(open('config.yaml', 'r'))
    with open('temporal_config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    command = [
        'python', 'train.py',
        '-model_path', './model/experiment',
        '-data', './monthly_20d',
        '-output', './output/experiment',
        '-ckpt_path', './ckpt/experiment',
        '-config_path', 'temporal_config.yaml',
        '-model_name', f'baseline_I20R{ret_days}',
        '-device', 'cuda',
        '-batch_size', '128',
        '-num_workers', '4',
        '-learning_rate', '0.001',
        '-num_epochs', '50',
        '-ret_days', f'{ret_days}',
        '-year_start', '1993',
        '-year_split', '1999',
        '-year_end', '2019',
        '-ratio', '0.7'
        ]
    subprocess.run(command)
    os.remove('temporal_config.yaml')
    for i in range(len(experiments)):
        name = f"{hyperparams[i]}_{experiments[i]}"
        config = yaml.safe_load(open('config.yaml', 'r'))
        config[hyperparams[i]] = experiments[i]
        with open('temporal_config.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        command = [
        'python', 'train.py',
        '-model_path', './model/experiment',
        '-data', './monthly_20d',
        '-output', './output/experiment',
        '-ckpt_path', './ckpt/experiment',
        '-config_path', 'temporal_config.yaml',
        '-model_name', name,
        '-device', 'cuda',
        '-batch_size', '128',
        '-num_workers', '4',
        '-learning_rate', '0.001',
        '-num_epochs', '50',
        '-ret_days', f'{ret_days}',
        '-year_start', '1993',
        '-year_split', '1999',
        '-year_end', '2019',
        '-ratio', '0.7'
        ]
        subprocess.run(command)
        os.remove('temporal_config.yaml')
def eval_exp(experiments, hyperparams, ret_days=5, input_dir='./model/experiment', output_dir='./output/experiment'):
    config = yaml.safe_load(open('config.yaml', 'r'))
    with open('temporal_config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    command = [
        'python', 'eval.py',
        '-model_path', input_dir,
        '-output', output_dir,
        '-config_path', 'temporal_config.yaml',
        '-model_name', f'baseline_I20R{ret_days}',
        '-data', './monthly_20d',
        '-device', 'cuda',
        '-batch_size', '128',
        '-num_workers', '4',
        '-ret_days', f'{ret_days}',
    ]
    subprocess.run(command)
    os.remove('temporal_config.yaml')
    for i in range(len(experiments)):
        name = f"{hyperparams[i]}_{experiments[i]}"
        config = yaml.safe_load(open('config.yaml', 'r'))
        config[hyperparams[i]] = experiments[i]
        with open('temporal_config.yaml', 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        command = [
            'python', 'eval.py',
            '-model_path', input_dir,
            '-output', output_dir,
            '-config_path', 'temporal_config.yaml',
            '-model_name', name,
            '-data', './monthly_20d',
            '-device', 'cuda',
            '-batch_size', '128',
            '-num_workers', '4',
            '-ret_days', f'{ret_days}',
        ]
        subprocess.run(command)
        os.remove('temporal_config.yaml')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments')
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Train the model')
    mode_group.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--ret_days', type=int, default=5, help='Number of days to predict')
    if '--train' in sys.argv:
        parser.add_argument('--output_dir', type=str, default='./model/experiment', help='Output directory')
    if '--eval' in sys.argv:
        parser.add_argument('--output_dir', type=str, default='./output/experiment', help='Output directory')
        parser.add_argument('--input_dir', type=str, default='./model/experiment', help='Input model directory')
    args = parser.parse_args()
    keys, values = zip(*experiment_variations.items())
    experiments = []
    hyperparams = []
    for key, value in zip(keys, values):
        experiments.extend(value)
        hyperparams.extend([key] * len(value))
    if args.train:
        train_exp(experiments, hyperparams, args.ret_days, args.output_dir)
    if args.eval:
        eval_exp(experiments, hyperparams, args.ret_days, args.input_dir, args.output_dir)

