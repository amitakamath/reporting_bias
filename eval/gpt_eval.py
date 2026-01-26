
import pdb
import json
import base64
import argparse
from tqdm import tqdm
from openai import OpenAI

api_key="PUT API KEY HERE"
client = OpenAI(api_key=api_key)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str,
                            choices=["spatial", \
                                    "negation", "temporal"])
    parser.add_argument("--model-name", default="4o",
            type=str, choices=["4o", "o1", "4V"])
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


def send_message(model, image, question):
    try:
        return client.chat.completions.create(
            model=model,
            max_completion_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}"},
                        },
                    ],
                }
            ],
        )
    except:
        raise


def main(args):
    # Load model
    if args.model_name == '4o':
        model = "gpt-4o-2024-08-06"
    elif args.model_name == 'o1':
        model = "o1-2024-12-17"
    elif args.model_name == 'o1-mini':
        model = "o1-mini-2024-09-12"
    elif args.model_name == '4V':
        model = "gpt-4-turbo-2024-04-09"
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
    no_pred_counter = 0
    preds = []

    for d in tqdm(data):
        question = build_conversation(args, d)
        image = encode_image('data/'+d['filename'])
        
        try:
            message = send_message(model, image, question)
        except:
            pdb.set_trace()
            print()
        
        pred = message.choices[0].message.content.strip()
        preds.append(pred)
        # Evaluate prediction
        if args.task == 'temporal' and len(set(pred).intersection(set(['A', 'B']))) != 1:
                print("No identified prediction?")
                print("Prediction: {}".format(pred))
                no_pred_counter += 1
        elif args.task in ['spatial', 'negation'] and \
                len(set(pred).intersection(set(['A', 'B', 'C', 'D']))) != 1:
            #pdb.set_trace()
            print("No identified prediction?")
            print("Prediction: {}".format(pred))
            no_pred_counter += 1
        else:
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
    json.dump(preds, open('predictions_{}_{}.json'.format(args.model_name, args.task), 'w'))
    print(correct * 100 / total)
    print(no_pred_counter*100/total)
    print()
        

if __name__ == "__main__":
    args = config()
    main(args)


