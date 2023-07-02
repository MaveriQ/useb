from useb import run, run_on
from transformers import AutoTokenizer, AutoModel
import torch
import argparse
from morphpiece import MorphPieceBPE
import pickle
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='maveriq/morphgpt-base-200k')
    parser.add_argument('--task', type=str, default='TwitterPara', choices=['all', 'AskUbuntu', 'CQADupStack', 'TwitterPara', 'SciDocs'])
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    return args

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = args.model.split('/')[-1]
    outfile = f'{args.output_dir}/{model_dir}-{args.task}.pkl'
    if not args.overwrite:
        assert not os.path.exists(outfile), f'{outfile} already exists. Use --overwrite to overwrite.'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'morph' in args.model:
        tokenizer = MorphPieceBPE(model_max_length=512)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token_id is None:
        print('Setting pad_token_id to eos_token_id')
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModel.from_pretrained(args.model).to(device)

    @torch.no_grad()
    def semb_fn(sentences) -> torch.Tensor:
        enc = tokenizer(sentences,return_tensors='pt',truncation=True,padding=True)
        enc = {k:v.to(device) for k,v in enc.items()}
        output = model(**enc)

        return (output.last_hidden_state.permute(0,2,1)*enc['attention_mask'].bool().unsqueeze(1)).mean(2)

    print(f"Evaluating {args.model} using {device} on {args.task}\n")

    if args.task == 'all':
        results, results_main_metric = run(
                semb_fn_askubuntu=semb_fn, 
                semb_fn_cqadupstack=semb_fn,  
                semb_fn_twitterpara=semb_fn, 
                semb_fn_scidocs=semb_fn,
                eval_type='valid',
                data_eval_path='data-eval'  # This should be the path to the folder of data-eval
            )
        print(results_main_metric)
    else:   
        results = run_on(args.task,semb_fn, 'valid')

    if args.save_results:
        with open(outfile, 'wb') as f:
            pickle.dump(results, f)

if __name__ == '__main__':      
    args = parse_args()

    if args.task == 'all':
        tasks = ['AskUbuntu', 'CQADupStack', 'TwitterPara', 'SciDocs']
        for task in tasks:
            args.task = task
            main(args)
    else:
        main(args)