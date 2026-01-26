
import pdb
import json
import base64
import argparse
import google.generativeai as genai
from tqdm import tqdm

genai.configure(api_key="PUT API KEY HERE")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str,
                            choices=["counting"])
    parser.add_argument("--model-name", default="1.5-flash",
            type=str, choices=["1.5-flash", "1.5-pro"])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--output-dir", default="./outputs", type=str)
    return parser.parse_args()


def build_conversation(args, d):
    options = d['caption_options']
    noun = options[0][2:]
    question = "How many {} are there in this picture? Answer with the number only, from 2-10.".format(noun)
    return question


def send_message(model_name, image, question):
    model = genai.GenerativeModel(model_name)
    try:
        response = model.generate_content([{'mime_type':'image/jpeg', 
            'data': image}, question])
        return response
    except:
        pdb.set_trace()
        print("what happened?")
        raise


def main(args):
    # Load model
    if args.model_name == '1.5-flash':
        model = "gemini-1.5-flash"
    elif args.model_name == '1.5-pro':
        model = "gemini-1.5-pro"
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

        pred = message.text.strip()
        
        # Evaluate prediction
        try:
            pred = int(pred)
            preds.append(pred)
            if pred == d['gold_index']+2:
                correct += 1
        except:
            if str(d['gold_index']+2) in pred:
                correct += 1
            print("Prediction wasn't a number")
            print("Pred: ")
            print(pred)
            print("Gold: ")
            print(d['gold_index']+2)
            print()
            preds.append(pred)

        total += 1
 

    pdb.set_trace()
    json.dump(preds, open('predictions_{}_{}.json'.format(args.model_name, args.task), 'w'))
    print(correct * 100 / total)
    print(no_pred_counter*100/total)
    print()
        

if __name__ == "__main__":
    args = config()
    main(args)


