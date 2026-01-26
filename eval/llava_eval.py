
import pdb
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration



def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str,
                            choices=["spatial", \
                                    "negation", "temporal"])
    parser.add_argument("--model-name", default="1.5-7b",
            type=str, choices=["1.5-7b", "1.5-13b", 
                "v1.6-vicuna-7b", "v1.6-vicuna-13b",
                "v1.6-mistral-7b", "v1.6-34b"])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--output-dir", default="./outputs", type=str)
    return parser.parse_args()


def build_conversation(args, d):
    options = d['caption_options']
    if args.task == 'temporal':
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", 
                     "text": "Pick the best caption for this image " \
                        "from the below options. Answer in one word, the " \
                        "option letter only. "\
                        "\n(A) {}\n(B) {}".format(
                        options[0].lower(), options[1].lower())},
                ],
            },
        ]
    else:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", 
                     "text": "Pick the best caption for this image " \
                        "from the below options. Answer in one word, the " \
                        "option letter only. "\
                        "\n(A) {}\n(B) {}\n(C) {}\n(D) {}".format(
                        options[0].lower(), options[1].lower(),
                        options[2].lower(), options[3].lower())},
                ],
            },
        ]
    return conversation


def main(args):
    # Load model
    if '1.5' in args.model_name:
        model = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-{}-hf".format(args.model_name), 
                torch_dtype=torch.float16, 
                device_map="auto")
        processor = AutoProcessor.from_pretrained(
                "llava-hf/llava-{}-hf".format(args.model_name)) 
    else:
       model = LlavaNextForConditionalGeneration.from_pretrained(
               "llava-hf/llava-{}-hf".format(args.model_name), 
               torch_dtype=torch.float16, 
               low_cpu_mem_usage=True, device_map="auto") 
       processor = LlavaNextProcessor.from_pretrained(
               "llava-hf/llava-{}-hf".format(args.model_name))

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
        prompt = processor.apply_chat_template(
                    conversation, add_generation_prompt=True)
        image = Image.open('data/'+d['filename'])
        inputs = processor(images=[image], text=[prompt], 
                    padding=True, return_tensors="pt").to(
                            model.device, torch.float16)
        generate_ids = model.generate(**inputs, max_new_tokens=30)
        output = processor.batch_decode(
                    generate_ids, skip_special_tokens=True)[0]
        if '1.5' in args.model_name or 'vicuna' in args.model_name:
            pred_index = output.index('ASSISTANT: ') + len('ASSISTANT: ')
        elif 'mistral' in args.model_name:
            pred_index = output.index('[/INST] ') + len('[/INST] ')
        else:
            raise NotImplementedError
        pred = output[pred_index:].strip()
        
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

    #pdb.set_trace()
    print(correct * 100 / total)
    print()
        

if __name__ == "__main__":
    args = config()
    main(args)


