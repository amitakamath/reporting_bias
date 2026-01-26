
import pdb
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig



def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str,
                            choices=["spatial", \
                                    "negation", "temporal"])
    parser.add_argument("--model-name", default="7B-D",
            type=str, choices=["7B-D", "7B-O", "72B"])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--output-dir", default="./outputs", type=str)
    return parser.parse_args()


def build_conversation(args, d):
    options = d['caption_options']
    if args.task == 'temporal':
        question = "Pick the best caption for this image " \
                    "from the below options. Answer in one word, the " \
                    "option letter only. "\
                    "\n(A) {}\n(B) {}".format(
                    options[0].lower(), options[1].lower())
    else:
        question = "Pick the best caption for this image " \
                    "from the below options. Answer in one word, the " \
                    "option letter only. "\
                    "\n(A) {}\n(B) {}\n(C) {}\n(D) {}".format(
                    options[0].lower(), options[1].lower(),
                    options[2].lower(), options[3].lower())
    return question


def main(args):
    # Load model
    processor = AutoProcessor.from_pretrained(
                'allenai/Molmo-{}-0924'.format(args.model_name),
                    trust_remote_code=True,
                        torch_dtype='auto',
                            device_map='auto'
                            )
    model = AutoModelForCausalLM.from_pretrained(
                'allenai/Molmo-{}-0924'.format(args.model_name),
                    trust_remote_code=True,
                        torch_dtype='auto',
                            device_map='auto'
                            )
    # This loads pretrained 7B-O weights, which are bad at QA.
    #custom_weights = torch.load("data/models/Molmo-7B-O-0924-Pretrained/updated_model.pt")
    #model.load_state_dict(custom_weights)

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
        conversation = build_conversation(args, d)
        inputs = processor.process(images=[Image.open('data/'+d['filename'])], text=conversation)
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        output = model.generate_from_batch(
                    inputs,
                        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                            tokenizer=processor.tokenizer
                            )
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        pred = generated_text

        # Evaluate prediction
        if args.task == 'temporal':
            if len(set(pred).intersection(set(['A', 'B']))) != 1:
                pdb.set_trace()
                print("No identified prediction?")
        elif args.task in ['spatial', 'negation'] and \
                len(set(pred).intersection(set(['A', 'B', 'C', 'D']))) != 1:
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

    #pdb.set_trace()
    print(correct * 100 / total)
    print()
        

if __name__ == "__main__":
    args = config()
    main(args)


