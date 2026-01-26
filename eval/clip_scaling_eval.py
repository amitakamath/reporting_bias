
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
    parser.add_argument("--device", default="cuda", type=str)
    return parser.parse_args()


def main(args):
    """
    list_of_models = [
            'openai',
            'Model-B-32_Data-80M_Samples-3B_lr-5e-4_bs-32k.pt',
            'Model-B-32_Data-80M_Samples-13B_lr-5e-4_bs-32k.pt',
            'Model-B-32_Data-80M_Samples-34B_lr-1e-3_bs-88k.pt',
            'Model-B-32_Data-400M_Samples-3B_lr-1e-3_bs-88k.pt',
            'Model-B-32_Data-400M_Samples-13B_lr-1e-3_bs-86k.pt',
            'Model-B-32_Data-400M_Samples-34B_lr-5e-4_bs-32k.pt',
            'Model-B-32_Data-2B_Samples-3B_lr-1e-3_bs-88k.pt',
            'Model-B-32_Data-2B_Samples-13B_lr-5e-4_bs-32k.pt',
            'Model-B-32_Data-2B_Samples-34B_lr-1e-3_bs-79k.pt']
    base_name = 'ViT-B-32'
    
    list_of_models = [
            'openai',
            'Model-B-16_Data-80M_Samples-3B_lr-1e-3_bs-88k.pt',
            'Model-B-16_Data-80M_Samples-13B_lr-1e-3_bs-88k.pt',
            'Model-B-16_Data-80M_Samples-34B_lr-1e-3_bs-88k.pt',
            'Model-B-16_Data-400M_Samples-3B_lr-1e-3_bs-88k.pt',
            'Model-B-16_Data-400M_Samples-13B_lr-5e-4_bs-33k.pt',
            'Model-B-16_Data-400M_Samples-34B_lr-1e-3_bs-88k.pt',
            'Model-B-16_Data-2B_Samples-3B_lr-1e-3_bs-88k.pt',
            'Model-B-16_Data-2B_Samples-13B_lr-1e-3_bs-88k.pt',
            'Model-B-16_Data-2B_Samples-34B_lr-1e-3_bs-88k.pt']
    base_name = 'ViT-B-16'
    
    list_of_models = [
            'openai',
            'Model-L-14_Data-80M_Samples-3B_lr-1e-3_bs-88k.pt',
            'Model-L-14_Data-80M_Samples-13B_lr-1e-3_bs-88k.pt',
            'Model-L-14_Data-80M_Samples-34B_lr-1e-3_bs-88k.pt',
            'Model-L-14_Data-400M_Samples-3B_lr-1e-3_bs-88k.pt',
            'Model-L-14_Data-400M_Samples-13B_lr-1e-3_bs-86k.pt',
            'Model-L-14_Data-400M_Samples-34B_lr-1e-3_bs-86k.pt',
            'Model-L-14_Data-2B_Samples-3B_lr-1e-3_bs-88k.pt',
            'Model-L-14_Data-2B_Samples-13B_lr-1e-3_bs-86k.pt',
            'Model-L-14_Data-2B_Samples-34B_lr-1e-3_bs-86k.pt']
    base_name = 'ViT-L-14'
    
    list_of_models = [
            'Model-g-14_Data-2B_Samples-13B_lr-5e-4_bs-64k.pt']
    base_name = 'ViT-g-14'
    """
    list_of_models = [
            'Model-H-14_Data-2B_Samples-34B_lr-5e-4_bs-79k.pt']
    base_name = 'ViT-H-14'
    results = []


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


    for model_name in list_of_models:
        if model_name == 'openai':
            model, _, preprocess = open_clip.create_model_and_transforms(
                    base_name, pretrained=model_name,
                    device=args.device)
            tokenizer = open_clip.get_tokenizer(base_name)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                    base_name, pretrained='scaling-laws-openclip/models--laion--scaling-laws-openclip/snapshots/ff857939164def1f6a27e2403ad5b978e1a8f839/{}'.format(model_name), device=args.device)
            tokenizer = open_clip.get_tokenizer(base_name)

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

        acc = correct / total
        print()
        print(model_name)
        print(acc)
        print() 
        results.append(acc)

    json.dump(results, open('clip_predictions/{}_{}.json'.format(base_name, args.task), 'w'))
    print()

if __name__ == "__main__":
    args = config()
    main(args)

