import argparse
import torch
import os
import json
import jieba
import nltk
from data.dataprocess import tokenize_en, tokenize_zh, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_IDX, BOS_IDX, EOS_IDX
from models.rnn_nmt import create_rnn_model
from models.transformer_nmt import create_transformer_model
from models.t5_nmt import create_t5_model
import config

def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def text_to_indices(tokens, vocab):
    return [vocab.get(token, vocab.get(UNK_TOKEN)) for token in tokens]

def main():
    parser = argparse.ArgumentParser(description="Inference script for NMT models")
    parser.add_argument('--sentence', type=str, required=True, help="Input sentence to translate")
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--model_type', type=str, default='rnn', choices=['rnn', 'transformer', 't5'])
    parser.add_argument('--src_lang', type=str, default='en', choices=['en', 'zh'], help="Source language (default: en)")
    parser.add_argument('--tgt_lang', type=str, default='zh', choices=['en', 'zh'], help="Target language (default: zh)")
    
    # Model configuration args
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
    
    # Load Vocabs
    if not os.path.exists('data/en_vocab.json') or not os.path.exists('data/zh_vocab.json'):
        print("Error: Vocab files not found.")
        return
        
    en_vocab = load_vocab('data/en_vocab.json')
    zh_vocab = load_vocab('data/zh_vocab.json')
    
    # Determine Source and Target Vocabs
    if args.src_lang == 'en':
        src_vocab = en_vocab
        tgt_vocab = zh_vocab
        tokenize_fn = tokenize_en
    else:
        src_vocab = zh_vocab
        tgt_vocab = en_vocab
        tokenize_fn = tokenize_zh
        
    idx2word = {v: k for k, v in tgt_vocab.items()}
    
    # Initialize Model
    print(f"Initializing {args.model_type} model...")
    if args.model_type == 'rnn':
        model_config = config.RNN_CONFIG.copy()
        model_config['attention_type'] = args.attention_type
        model = create_rnn_model(len(src_vocab), len(tgt_vocab), model_config)
    elif args.model_type == 'transformer':
        model_config = config.TRANSFORMER_CONFIG.copy()
        model_config['position_encoding'] = args.position_encoding
        model_config['norm_type'] = args.norm_type
        if args.d_model is not None: model_config['d_model'] = args.d_model
        if args.nhead is not None: model_config['nhead'] = args.nhead
        if args.num_encoder_layers is not None: model_config['num_encoder_layers'] = args.num_encoder_layers
        if args.num_decoder_layers is not None: model_config['num_decoder_layers'] = args.num_decoder_layers
        if args.dim_feedforward is not None: model_config['dim_feedforward'] = args.dim_feedforward
        model = create_transformer_model(len(src_vocab), len(tgt_vocab), model_config)
    elif args.model_type == 't5':
        model_config = config.T5_CONFIG.copy()
        model = create_t5_model(model_config['model_name'], device=device)
        
    model = model.to(device)
    model.eval()
    
    # Load Checkpoint
    print(f"Loading checkpoint from {args.ckpt_path}...")
    if os.path.exists(args.ckpt_path):
        state_dict = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Error: Checkpoint file {args.ckpt_path} not found.")
        return
        
    # Preprocess Input
    print(f"Input Sentence: {args.sentence}")
    tokens = tokenize_fn(args.sentence)
    print(f"Tokens: {tokens}")
    
    if args.model_type == 't5':
        # T5 handles tokenization internally
        with torch.no_grad():
            generated_ids = model([args.sentence])
            decoded_preds = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            print(f"Translation: {decoded_preds[0]}")
            return

    # Convert to indices
    indices = [BOS_IDX] + text_to_indices(tokens, src_vocab) + [EOS_IDX]
    src_tensor = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0) # [1, seq_len]
    
    # Inference
    print("Translating...")
    with torch.no_grad():
        if args.model_type == 'rnn':
            src_mask = (src_tensor != PAD_IDX).float()
            decoded_ids, _ = model.greedy_decode(src_tensor, src_mask, max_len=50)
        elif args.model_type == 'transformer':
            decoded_ids = model.greedy_decode(src_tensor, max_len=50)
            
    # Decode Output
    decoded_ids = decoded_ids.squeeze(0).cpu().numpy()
    output_tokens = []
    for idx in decoded_ids:
        if idx == EOS_IDX:
            break
        if idx not in [BOS_IDX, PAD_IDX]:
            output_tokens.append(idx2word.get(idx, '<unk>'))
            
    if args.tgt_lang == 'zh':
        translation = "".join(output_tokens)
    else:
        translation = " ".join(output_tokens)
        
    print(f"Translation: {translation}")

if __name__ == "__main__":
    main()
