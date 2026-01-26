
import pdb
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str,
                            choices=["counting"])
    parser.add_argument("--model-name", default="1-7b",
            type=str, choices=["1-7b",])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--output-dir", default="./outputs", type=str)
    return parser.parse_args()


def build_conversation(args, d):
    options = d['caption_options']
    noun = options[0][2:]
    question = "How many {} are there in this picture? Answer with the number only, from 2-10.".format(noun)
    query = [
        {'image': 'data/'+d['filename']},
        {'text': question},
    ]
    return query


def main(args):
    # Load model
    if '1-7b' in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", 
                trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", 
                device_map="cuda", trust_remote_code=True).eval()
    else:
        raise NotImplementedError

    # Load eval data
    if args.task == 'counting':
        data = json.load(open('data/count_bench_reformatted.json'))
    else:
        raise NotImplementedError

    # Evaluate the model
    correct = 0
    total = 0
    preds = []
    null_count = 0
    for d in tqdm(data):
        query = build_conversation(args, d)
        query = tokenizer.from_list_format(query)
        response, history = model.chat(tokenizer, query=query, history=None)
        pred = response.strip()
        try:
            pred = int(pred)
            if pred == d['gold_index']+2:
                correct += 1
        except:
            if str(d['gold_index']+2) in response.strip():
                correct += 1
            else:
                print()
                print("Prediction wasn't a number")
                print(pred)
                print()
                null_count += 1
        """
        inputs = tokenizer(query, return_tensors='pt')
        inputs = inputs.to(model.device)
        pred = model.generate(**inputs)
        response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        print(response)
        pdb.set_trace()
        """
        total += 1

    pdb.set_trace()
    print(correct * 100 / total)
    print(null_count)
    print()
        

if __name__ == "__main__":
    args = config()
    main(args)


