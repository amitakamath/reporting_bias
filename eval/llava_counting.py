
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
                            choices=["counting"])
    parser.add_argument("--model-name", default="1.5-7b",
            type=str, choices=["1.5-7b", "1.5-13b", 
                "v1.6-vicuna-7b", "v1.6-vicuna-13b",
                "v1.6-mistral-7b", "v1.6-34b"])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--output-dir", default="./outputs", type=str)
    return parser.parse_args()


def build_conversation(args, d):
    options = d['caption_options']
    noun = options[0][2:]
    question = "How many {} are there in this picture? Answer with the number only, from 2-10.".format(noun)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", 
                 "text": question},
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
    if args.task == 'counting':
        data = json.load(open('data/count_bench_reformatted.json'))
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
        try:
            pred = int(pred)
            if pred == d['gold_index']+2:
                correct += 1
        except:
            print()
            print("Prediction wasn't a number")
            print(pred)
            print()

        total += 1

    #pdb.set_trace()
    print(correct * 100 / total)
    print()
        

if __name__ == "__main__":
    args = config()
    main(args)


