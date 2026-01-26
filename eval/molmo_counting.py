
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
                            choices=["counting"])
    parser.add_argument("--model-name", default="7B-D",
            type=str, choices=["7B-D", "7B-O", "72B"])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--output-dir", default="./outputs", type=str)
    return parser.parse_args()


def build_conversation(args, d):
    options = d['caption_options']
    noun = options[0][2:]
    question = "How many {} are there in this picture? Answer with the number only, from 2-10.".format(noun)
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

    # Load eval data
    if args.task == 'counting':
        data = json.load(open('data/count_bench_reformatted.json'))
    else:
        raise NotImplementedError

    # Evaluate the model
    correct = 0
    total = 0
    preds = []
    correct_dict = {k: 0 for k in range(2, 11)}
    total_dict = {k: 0 for k in range(2, 11)}

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
        try:
            pred = int(generated_text)
            if pred == d['gold_index']+2:
                correct += 1
                correct_dict[d['gold_index']+2] += 1
        except:
            print("Prediction wasn't a number")
        
        total += 1
        total_dict[d['gold_index']+2] += 1

    pdb.set_trace()
    print(correct * 100 / total)
    print()
        

if __name__ == "__main__":
    args = config()
    main(args)


