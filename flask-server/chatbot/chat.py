import random
import re
import torch
from nltk_utils import vietnamese_tokenizer, bag_of_words

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
