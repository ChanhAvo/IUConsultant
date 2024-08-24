import random
import json 
import torch 
import re
from model import NeuralNet 
from nltk_utils import vietnamese_tokenizer, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('flask-server/resources/Intents.json', 'r') as f:
    intents = json.load(f)
with open('flask-server/resources/Questions.json', 'r') as f:
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

while True:
    sentence = input('Bạn: ')
    if sentence == "Tôi không muốn trò chuyện":
        break 
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
                        if re.search(pattern, sentence):
                            answer = q_a['answer']
                            print(f"{bot_name}: {answer}")
                            matched = True
                            break
                else:
                    if re.search(q_a['question'], sentence):
                        answer = q_a['answer']
                        print(f"{bot_name}: {answer}")
                        matched = True
                        break
            if matched:
                break
                
        if not matched:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
                    break  
    else:
        print(f"{bot_name}: Mình chưa hiểu ý của bạn...")
