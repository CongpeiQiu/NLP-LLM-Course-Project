import argparse
import torch
import json
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import config
from data.dataprocess import load_and_process_data, TranslationDataset, collate_fn, TEST_PATH, VALID_PATH, PAD_IDX
from models.rnn_nmt import create_rnn_model
from models.transformer_nmt import create_transformer_model
from models.t5_nmt import create_t5_model
from train import evaluate, load_vocab

def main():
    parser = argparse.ArgumentParser(description="Evaluate NMT model with Beam Search")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--data_path', type=str, default=TEST_PATH, help="Path to data file (default: test set)")
    parser.add_argument('--decoding_strategy', type=str, default='greedy', choices=['greedy', 'beam'], help="Decoding strategy")
    parser.add_argument('--beam_size', type=int, default=3, help="Beam size for decoding")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--output_file', type=str, default='evaluation_results.json', help="File to save evaluation results")
    
    # Model configuration args (must match training)
    parser.add_argument('--model_type', type=str, default='rnn', choices=['rnn', 'transformer', 't5'])
    parser.add_argument('--attention_type', type=str, default='additive', choices=['dot', 'multiplicative', 'additive'])
    parser.add_argument('--position_encoding', type=str, default='absolute', choices=['absolute', 'relative', 'none'])
    parser.add_argument('--norm_type', type=str, default='layernorm', choices=['layernorm', 'rmsnorm'])
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--nhead', type=int, default=None)
    parser.add_argument('--num_encoder_layers', type=int, default=None)
    parser.add_argument('--num_decoder_layers', type=int, default=None)
    parser.add_argument('--dim_feedforward', type=int, default=None)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Vocab
    if not os.path.exists('data/en_vocab.json') or not os.path.exists('data/zh_vocab.json'):
        print("Error: Vocab files not found.")
        return
    
    en_vocab = load_vocab('data/en_vocab.json')
    zh_vocab = load_vocab('data/zh_vocab.json')

    # Load Data
    print(f"Loading data from {args.data_path}...")
    # Use is_test=True to avoid filtering out valid test cases if any logic exists
    test_data = load_and_process_data(args.data_path, is_test=True)
    test_dataset = TranslationDataset(test_data, en_vocab, zh_vocab)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize Model
    print(f"Initializing {args.model_type} model...")
    if args.model_type == 'rnn':
        model_config = config.RNN_CONFIG.copy()
        model_config['attention_type'] = args.attention_type
        model = create_rnn_model(len(en_vocab), len(zh_vocab), model_config)
    elif args.model_type == 'transformer':
        model_config = config.TRANSFORMER_CONFIG.copy()
        model_config['position_encoding'] = args.position_encoding
        model_config['norm_type'] = args.norm_type
        if args.d_model is not None: model_config['d_model'] = args.d_model
        if args.nhead is not None: model_config['nhead'] = args.nhead
        if args.num_encoder_layers is not None: model_config['num_encoder_layers'] = args.num_encoder_layers
        if args.num_decoder_layers is not None: model_config['num_decoder_layers'] = args.num_decoder_layers
        if args.dim_feedforward is not None: model_config['dim_feedforward'] = args.dim_feedforward
        model = create_transformer_model(len(en_vocab), len(zh_vocab), model_config)
    elif args.model_type == 't5':
        model_config = config.T5_CONFIG.copy()
        model = create_t5_model(model_config['model_name'], device=device)

    model = model.to(device)

    # Load Checkpoint
    print(f"Loading checkpoint from {args.ckpt_path}...")
    if os.path.exists(args.ckpt_path):
        state_dict = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Checkpoint file {args.ckpt_path} not found.")
        return

    # Evaluate
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    print(f"Starting evaluation with strategy '{args.decoding_strategy}' (beam_size={args.beam_size})...")
    
    loss, bleu = evaluate(
        model, test_loader, criterion, device, zh_vocab, 
        decoding_strategy=args.decoding_strategy, beam_size=args.beam_size
    )

    print(f"\nEvaluation Results:")
    print(f"Loss: {loss:.4f}")
    print(f"BLEU: {bleu:.2f}")

    # Save results
    result = {
        'ckpt_path': args.ckpt_path,
        'model_type': args.model_type,
        'decoding_strategy': args.decoding_strategy,
        'beam_size': args.beam_size,
        'loss': loss,
        'bleu': bleu
    }

    if args.model_type == 'rnn':
        result['attention_type'] = args.attention_type
    elif args.model_type == 'transformer':
        result['position_encoding'] = args.position_encoding
        result['norm_type'] = args.norm_type
        # Use args if provided, otherwise fallback to config defaults
        defaults = config.TRANSFORMER_CONFIG
        result['d_model'] = args.d_model if args.d_model is not None else defaults.get('d_model')
        result['nhead'] = args.nhead if args.nhead is not None else defaults.get('nhead')
        result['num_encoder_layers'] = args.num_encoder_layers if args.num_encoder_layers is not None else defaults.get('num_encoder_layers')
        result['num_decoder_layers'] = args.num_decoder_layers if args.num_decoder_layers is not None else defaults.get('num_decoder_layers')
        result['dim_feedforward'] = args.dim_feedforward if args.dim_feedforward is not None else defaults.get('dim_feedforward')
    
    # Append to list in JSON file or create new
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r') as f:
                results = json.load(f)
                if not isinstance(results, list):
                    results = [results]
        except json.JSONDecodeError:
            results = []
    else:
        results = []
        
    results.append(result)
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
