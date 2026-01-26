
import pdb
import json
import clip
import torch
import random
import argparse
import open_clip
from PIL import Image
from tqdm import tqdm


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str,
                            choices=["spatial", "counting", \
                                    "negation", "temporal"])
    parser.add_argument("--model-name", default="openai-clip:ViT-B/32",
            type=str, choices=["openai-clip:ViT-B/32", "openai-clip:ViT-B/16",
                "openai-clip:ViT-L/14",
                "laion-clip:ViT-B/32", "laion-clip:ViT-B/16", 
                "laion-clip:ViT-L/14", "laion-clip:ViT-g/14",
                "laion-clip:ViT-H/14",
                "thao-clip:ViT-B/32"])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--output-dir", default="./outputs", type=str)
    return parser.parse_args()


def main(args):
    # Load model
    if args.model_name == 'openai-clip:ViT-B/32':
        model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai', device=args.device)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
    elif args.model_name == 'openai-clip:ViT-B/16':
        model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-16', pretrained='openai', device=args.device)
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
    elif args.model_name == 'openai-clip:ViT-L/14':
        model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-L-14', pretrained='openai', device=args.device)
        tokenizer = open_clip.get_tokenizer('ViT-L-14')

    elif args.model_name == 'laion-clip:ViT-B/32':
        model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k', 
                device=args.device)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
    elif args.model_name == 'laion-clip:ViT-B/16':
        model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-16', pretrained='laion2b_s34b_b88k', 
                device=args.device)
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
    elif args.model_name == 'laion-clip:ViT-L/14':
        model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-L-14', pretrained='laion2b_s32b_b82k', 
                device=args.device)
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
    elif args.model_name == 'laion-clip:ViT-g/14':
        model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-g-14', pretrained='laion2b_s12b_b42k', 
                device=args.device)
        tokenizer = open_clip.get_tokenizer('ViT-g-14')
    elif args.model_name == 'laion-clip:ViT-H/14':
        model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-H-14', pretrained='laion2b_s32b_b79k', 
                device=args.device)
        tokenizer = open_clip.get_tokenizer('ViT-H-14')

    elif args.model_name == 'thao-clip:ViT-B/32':
        model, _, preprocess = open_clip.create_model_and_transforms(
                'hf-hub:thaottn/datacomp-medium_basic_DFN_filtered_f0.2_translated_captions_AND_original_captions', device=args.device)
        tokenizer = open_clip.get_tokenizer('hf-hub:thaottn/datacomp-medium_basic_DFN_filtered_f0.2_translated_captions_AND_original_captions')
    else:
        raise NotImplementedError

    # Load eval data
    if args.task == 'spatial':
        data = json.load(open('data/spatial.json'))
    elif args.task == 'counting':
        data = json.load(open('data/count_bench_reformatted.json'))
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
        image = preprocess(Image.open(
                        'data/'+d['filename'])).unsqueeze(0).to(args.device)
        text = tokenizer([c.lower() 
                    for c in d['caption_options']]).to(args.device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ 
                            text_features.T).softmax(dim=-1)[0]

        if torch.argmax(text_probs).item() == d['gold_index']:
            correct += 1
        total += 1
        #preds.append(d['caption_options'][torch.argmax(text_probs).item()])

    print()
    print("Accuracy:")
    print(correct * 100 / total)
    print()    


if __name__ == "__main__":
    args = config()
    main(args)

