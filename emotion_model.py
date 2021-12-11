import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification



def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained('bhadresh-savani/distilbert-base-uncased-emotion').to(device)
    tokenizer = AutoTokenizer.from_pretrained('bhadresh-savani/distilbert-base-uncased-emotion')

    return model,tokenizer

def get_emotion(text,model,tokenizer):
    
    def get_inputs(text):
        tokens = tokenizer(text)
        inputs = torch.tensor(tokens['input_ids']).unsqueeze(0)
        attention_mask = torch.tensor(tokens['attention_mask']).unsqueeze(0)
        return inputs, attention_mask


    def get_proba(output):
        return torch.exp(output)/torch.sum(torch.exp(output))

    def get_output(probs):
        out_dict = {
            "sadness": probs[0].item(),
            "joy": probs[1].item(),
            "love": probs[2].item(),
            "anger": probs[3].item(),
            "fear": probs[4].item(),
            "surprise": probs[5].item()
             }

        return out_dict 


    inputs, attention_mask = get_inputs(text)
    output  = model(inputs, attention_mask=attention_mask)
    probs = get_proba(output[0][0])
    emotions = get_output(probs)

    return emotions  

def dict_argmax(dict):
    max_val = -1
    for key in dict:
        if dict[key] >= max_val:
            argmax = key
            max_val = dict[key]

    return argmax