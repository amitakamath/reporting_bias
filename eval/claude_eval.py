
import pdb
import json
import base64
import argparse
import anthropic
from tqdm import tqdm

client = anthropic.Anthropic(api_key="PUT API KEY HERE")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode('utf-8')


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str,
                            choices=["spatial", \
                                    "negation", "temporal"])
    parser.add_argument("--model-name", default="sonnet",
            type=str, choices=["sonnet", "haiku"])
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
        return client.messages.create(
            model=model,
            max_tokens=20,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image,
                            },
                        },
                        {
                            "type": "text",
                             "text": question,
                        }
                    ],
                }
            ],
        )
    except:
        raise


def main(args):
    # Load model
    if args.model_name in ['sonnet', 'haiku']:
        model = "claude-3-5-{}-20241022".format(args.model_name)
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
        question = build_conversation(args, d)
        image = encode_image('data/'+d['filename'])
        try:
            message = send_message(model, image, question)
        except:
            pdb.set_trace()
            print()
         
        pred = message.content[0].text.strip()
        preds.append(pred)
        # Evaluate prediction
        if args.task == 'temporal' and len(set(pred).intersection(set(['A', 'B']))) != 1:
            print("No identified prediction?")
            print("Prediction: {}".format(pred))
            print("Answer: {}".format(d['gold_index']))
        elif args.task in ['spatial', 'negation'] and \
                len(set(pred).intersection(set(['A', 'B', 'C', 'D']))) != 1:
            print("No identified prediction?")
            print("Prediction: {}".format(pred))
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
    print()
        

if __name__ == "__main__":
    args = config()
    main(args)


