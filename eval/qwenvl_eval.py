
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
                            choices=["spatial", \
                                    "negation", "temporal"])
    parser.add_argument("--model-name", default="1-7b",
            type=str, choices=["1-7b",])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--output-dir", default="./outputs", type=str)
    return parser.parse_args()


def build_conversation(args, d):
    options = d['caption_options']
    if args.task == 'temporal':
        query = [
            {'image': 'data/'+d['filename']},
            {'text': "Pick the best caption for this image " \
                    "from the below options. Answer in one word, the " \
                    "option letter only. "\
                    "\n(A) {}\n(B) {}".format(
                        options[0].lower(), options[1].lower())},
        ]
    else:
        query = [
            {'image': 'data/'+d['filename']},
            {'text': "Pick the best caption for this image " \
                    "from the below options. Answer in one word, the " \
                    "option letter only. "\
                        "\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n" \
                        "Answer: ".format(
                        options[0].lower(), options[1].lower(),
                        options[2].lower(), options[3].lower())},
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
    if args.task == 'spatial':
        data = json.load(open('data/spatial.json'))
    elif args.task == 'temporal':
        data = json.load(open('data/temporal_data_reformatted.json'))
    elif args.task == 'negation':
        data = json.load(open('data/negation_val.json'))
    else:
        raise NotImplementedError

    # Evaluate the model
    correct = 0
    total = 0
    preds = []

    for d in tqdm(data):
        query = build_conversation(args, d)
        query = tokenizer.from_list_format(query)
        response, history = model.chat(tokenizer, query=query, history=None)
        pred = response.strip()
        """
        inputs = tokenizer(query, return_tensors='pt')
        inputs = inputs.to(model.device)
        pred = model.generate(**inputs)
        response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        print(response)
        pdb.set_trace()
        """
        # Evaluate prediction
        if len(set(pred).intersection(set(['A', 'B', 'C', 'D']))) != 1:
            pdb.set_trace()
            print("No identified prediction?")
        if d['gold_index'] == 0 and ('A' in pred or '(A)' in pred):
            correct += 1
        elif d['gold_index'] == 1 and ('B' in pred or '(B)' in pred):
            correct += 1
        elif d['gold_index'] == 2 and ('C' in pred or '(C)' in pred):
            correct += 1
        elif d['gold_index'] == 3 and ('D' in pred or '(D)' in pred):
            correct += 1
        total += 1

    pdb.set_trace()
    print(correct * 100 / total)
    print()
        

if __name__ == "__main__":
    args = config()
    main(args)


