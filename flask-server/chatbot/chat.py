import random
import json
import torch
import re
from model import NeuralNet
from nltk_utils import vietnamese_tokenizer, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('../resources/Intents.json', 'r') as f:
    intents = json.load(f)
with open('../resources/Questions.json', 'r') as f:
    questions = json.load(f)

all_questions = []
for intent in intents['intents']:
    all_questions.extend(intent['patterns'])
for question_dict in questions['questions']:
    if 'questions_and_answers' in question_dict:
        for q_a in question_dict['questions_and_answers']:
            if isinstance(q_a['question'], list):
                all_questions.extend(q_a['question'])
            else:
                all_questions.append(q_a['question'])

FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "IU Consultant"
print("Bắt đầu cuộc trò chuyện. Nếu như bạn chưa muốn bắt đầu, hãy type 'Tôi không muốn trò chuyện' để thoát cuộc trò chuyện")

def get_score(major, method):
    with open('../resources/Scores.json', 'r') as f:
        data = json.load(f)

    method_key = f"method{method}"
    if major not in data['major']:
        return f"Không tìm thấy ngành {major}"
    
    major_data = data['major'][major]

    if method_key in major_data:
        return major_data[method_key]
    elif 'method4' in major_data:
        for key in major_data['method4']:
            if method_key in key.lower():
                return major_data['method4'][key]
        return "Không tìm thấy phương thức xét tuyển phù hợp trong method4"
    else:
        return "Không tìm thấy phương thức xét tuyển"


def process_chatbot_response(sentence, model, all_words, tags, device, questions, intents, get_score, bot_name):
    sentence_tokenized = vietnamese_tokenizer(sentence)
    X = bag_of_words(sentence_tokenized, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        matched = False
        for question_tag in questions['questions']:
            for q_a in question_tag['questions_and_answers']:
                if isinstance(q_a['question'], list):
                    for pattern in q_a['question']:
                        match = re.search(pattern, sentence, re.IGNORECASE)
                        if match:
                            response = generate_response(q_a['answer'], match, get_score)
                            return response  # Return the response to Flask
                            matched = True
                            break
                else:
                    match = re.search(q_a['question'], sentence, re.IGNORECASE)
                    if match:
                        response = generate_response(q_a['answer'], match, get_score)
                        return response  # Return the response to Flask
                        matched = True
                        break
            if matched:
                break

        if not matched:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])  # Return matched response
    else:
        return "Mình chưa hiểu ý của bạn..."  # Return fallback response


def generate_response(answer_template, match, get_score):
    if "{major}" in answer_template and "{method}" in answer_template:
        major = match.group(1)
        method = match.group(2)
        score = get_score(major.strip(), method.strip())
        response = answer_template.replace("{major}", major).replace("{method}", method).replace("{score}", str(score))
    else:
        response = answer_template
        for i in range(1, len(match.groups()) + 1):
            response = re.sub(r"\(\.\+\?\)", match.group(i), response, 1)
    return response

while True:
    sentence = input('Bạn: ')
    if sentence == "Tôi không muốn trò chuyện":
        break
    process_chatbot_response(sentence, model, all_words, tags, device, questions, intents, get_score, bot_name)

