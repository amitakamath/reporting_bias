
import pdb
import json
import base64
import argparse
from tqdm import tqdm
from openai import OpenAI

api_key = "PUT API KEY HERE"
client = OpenAI(api_key=api_key)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str,
                            choices=["counting"])
    parser.add_argument("--model-name", default="4o",
            type=str, choices=["4o", "o1", "4V"])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--output-dir", default="./outputs", type=str)
    return parser.parse_args()


def build_conversation(args, d):
    options = d['caption_options']
    noun = options[0][2:]
    question = "How many {} are there in this picture? Answer with the number only, from 2-10.".format(noun)
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
    if args.task == 'counting':
        data = json.load(open('data/count_bench_reformatted.json'))
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
        try:
            pred = int(pred)
            preds.append(pred)
            if pred == d['gold_index']+2:
                correct += 1
        except:
            print("Prediction wasn't a number")
            preds.append(None)
            print("Prediction: {}".format(pred))
            print()
            no_pred_counter+=1
        total += 1
        

    pdb.set_trace()
    json.dump(preds, open('predictions_{}_{}.json'.format(args.model_name, args.task), 'w'))
    print(correct * 100 / total)
    print()
        

if __name__ == "__main__":
    args = config()
    main(args)


