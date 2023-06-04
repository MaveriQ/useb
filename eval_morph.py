from useb import run, run_on
from transformers import AutoTokenizer, AutoModel
import torch
import argparse
from morphpiece import MorphPieceBPE
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='bert-base-uncased')
parser.add_argument('--task', type=str, default='AskUbuntu', choices=['AskUbuntu', 'CQADupStack', 'TwitterPara', 'SciDocs'])
parser.add_argument('--output_dir', type=str, default='output')
parser.add_argument('--save_results', action='store_true')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
model_dir = args.model.split('/')[-1]
outfile = f'{args.output_dir}/{model_dir}-{args.task}.pkl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if 'morph' in args.model:
    tokenizer = MorphPieceBPE(model_max_length=512)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModel.from_pretrained(args.model).to(device)

@torch.no_grad()
def semb_fn(sentences) -> torch.Tensor:
    enc = tokenizer(sentences,return_tensors='pt',truncation=True,padding=True)
    enc = {k:v.to(device) for k,v in enc.items()}
    output = model(**enc)

    return (output.last_hidden_state.permute(0,2,1)*enc['attention_mask'].bool().unsqueeze(1)).mean(2)

print(f"Evaluating {args.model} using {device} on {args.task}\n")

results = run_on(args.task,semb_fn, 'valid')

if args.save_results:
    with open(outfile, 'wb') as f:
        pickle.dump(results, f)