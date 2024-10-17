from flask import Flask, render_template, request, jsonify
import torch, json 
from model import NeuralNet
from chat import process_chatbot_response  # Import chatbot logic here
from nltk_utils import vietnamese_tokenizer, bag_of_words

app = Flask(__name__)

# Load model and other components globally once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FILE = "data.pth"
data = torch.load(FILE)
model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"]).to(device)
model.load_state_dict(data["model_state"])
model.eval()

# Load intents, questions, and other data
all_words = data["all_words"]
tags = data["tags"]

with open('../resources/Intents.json', 'r') as f:
    intents = json.load(f)

with open('../resources/Questions.json', 'r') as f:
    questions = json.load(f)

# Function to retrieve scores
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
    # Return the response as JSON
    return jsonify({"answer": response})
 
if __name__ == '__main__':
    app.run(debug=True)
