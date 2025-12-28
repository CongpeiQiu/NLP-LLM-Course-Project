# Neural Machine Translation (RNN & Transformer)

This project implements Neural Machine Translation (NMT) models using RNN (LSTM) and Transformer architectures for Chinese-to-English translation tasks.

## 1. Training
Use `train.py` to train models. Training logs are saved as JSON files in `logs/`, and model checkpoints are saved in `checkpoints/`.

### Basic Usage
```bash
python3 train.py --model_type [rnn|transformer|t5] --exp_name [experiment_name]
```

### Example Commands
**Train RNN with Additive Attention:**
```bash
python3 train.py --model_type rnn --attention_type additive --exp_name rnn_add --epochs 30
```

**Train Transformer (Absolute PE, LayerNorm):**
```bash
python3 train.py --model_type transformer --position_encoding absolute --norm_type layernorm --exp_name trans_abs --epochs 30
```

### Key Arguments
- `--model_type`: Model architecture (`rnn`, `transformer`, `t5`).
- `--exp_name`: Unique name for the experiment (used for output filenames).
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size (default: 32).
- `--lr`: Learning rate.
- **RNN Specific**:
  - `--attention_type`: `dot`, `multiplicative`, `additive`.
  - `--teacher_forcing_ratio`: Probability of using teacher forcing (0.0 to 1.0).
- **Transformer Specific**:
  - `--position_encoding`: `absolute`, `relative`, `none`.
  - `--norm_type`: `layernorm`, `rmsnorm`.
  - `--d_model`, `--nhead`, `--num_encoder_layers`, etc.: Model dimensions.

## 2. Evaluation
Use `evaluate.py` to evaluate trained models on the test set using BLEU score and Loss. Supports Beam Search and Greedy Decoding.

```bash
python3 evaluate.py --ckpt_path checkpoints/rnn_add_best.pt --model_type rnn --attention_type additive --decoding_strategy beam --beam_size 3
```
Results are appended to `evaluation_results.json`.

## 3. Visualization
Use `visualize.py` to plot training loss and validation metrics from the generated log files.

### Single/Multi-Log Visualization
```bash
python3 visualize.py \
    --log_files logs/rnn_add_log.json logs/trans_abs_log.json \
    --labels "RNN Additive" "Transformer Abs" \
    --output_dir plots
```
