from flask import Flask, render_template, request, jsonify
import torch, json 
from model import NeuralNet
from chat import process_chatbot_response
from nltk_utils import vietnamese_tokenizer, bag_of_words

app = Flask(__name__)

# Assuming model and other components are initialized here as shown previously
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FILE = "data.pth"
data = torch.load(FILE)
model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"]).to(device)
model.load_state_dict(data["model_state"])
model.eval()

all_words = data["all_words"]
tags = data["tags"]

with open('../resources/Intents.json', 'r') as f:
    intents = json.load(f)

with open('../resources/Questions.json', 'r') as f:
    questions = json.load(f)

def get_score(major, method):
    with open('../resources/Scores.json', 'r') as f:
        scores = json.load(f)
    # Additional logic for score retrieval
    return scores

@app.route("/")
def index_get():
    return render_template("chat.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.get_json().get("message")

    if not text:
        return jsonify({"answer": "No message received."}), 400

    # Pass all required arguments
    response = process_chatbot_response(
        sentence=text,
        model=model,
        all_words=all_words,
        tags=tags,
        device=device,
        questions=questions,
        intents=intents,
        get_score=get_score,
        bot_name="IU Consultant"
    )
    
    return jsonify({"answer": response})

if __name__ == '__main__':
    app.run(debug=True)
