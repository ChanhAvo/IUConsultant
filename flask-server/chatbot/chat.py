import random
import json
import torch
import re
from model import NeuralNet
from nltk_utils import vietnamese_tokenizer, bag_of_words

# Load device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents and questions data
with open('../resources/Intents.json', 'r') as f:
    intents = json.load(f)
with open('../resources/Questions.json', 'r') as f:
    questions = json.load(f)

# Load the model and its data
FILE = "data.pth"
data = torch.load(FILE, weights_only=True)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Define bot name
bot_name = "IU Consultant"

# Greeting message
print("Bắt đầu cuộc trò chuyện. Nếu như bạn chưa muốn bắt đầu, hãy type 'Tôi không muốn trò chuyện' để thoát cuộc trò chuyện")

# Function to generate responses based on chatbot logic
def generate_response(answer_template, match, get_score):
    if "{major}" in answer_template and "{method}" in answer_template:
        major = match.group(1)
        method = match.group(2)
        score = get_score(major.strip(), method.strip())
        # Format the final response by replacing placeholders
        response = answer_template.replace("{major}", major).replace("{method}", method).replace("{score}", str(score))
    else:
        # Simple response generation without dynamic placeholders
        response = answer_template
        for i in range(1, len(match.groups()) + 1):
            response = re.sub(r"\(\.\+\?\)", match.group(i), response, 1)
    return response

# Main process to handle chatbot response logic
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
            # If no question is matched, check intents and return a random response
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])  # Return matched response
    else:
        return "Mình chưa hiểu ý của bạn..."  # Return fallback response if probability is too low

# Main loop to handle conversation
while True:
    sentence = input('Bạn: ')
    if sentence.lower() == "tôi không muốn trò chuyện":
        print(f"{bot_name}: Tạm biệt bạn! Hẹn gặp lại!")
        break

    response = process_chatbot_response(sentence, model, all_words, tags, device, questions, intents, get_score, bot_name)
    print(f"{bot_name}: {response}")
