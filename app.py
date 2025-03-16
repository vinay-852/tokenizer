from flask import Flask, request, jsonify, render_template
import os
import time
from src import BasicTokenizer, RegexTokenizer

app = Flask(__name__, template_folder='templates')
tokenizer = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    global tokenizer
    data = request.get_json()
    model_name = data.get("model_name", "basic")
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # create a directory for models, so we don't pollute the current directory
    os.makedirs("models", exist_ok=True)

    t0 = time.time()
    if model_name == "basic":
        tokenizer = BasicTokenizer()
    elif model_name == "regex":
        tokenizer = RegexTokenizer()
    else:
        return jsonify({"error": "Invalid model name"}), 400

    tokenizer.train(text, 512, verbose=True)
    t1 = time.time()

    return jsonify({"status": "Training completed", "time_taken": f"{t1 - t0:.2f} seconds"})

@app.route('/tokenize', methods=['POST'])
def tokenize():
    global tokenizer
    if tokenizer is None:
        return jsonify({"error": "Tokenizer not trained"}), 400
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    tokens = tokenizer.encode(text)
    return jsonify({"tokens": tokens})

@app.route('/token_info', methods=['POST'])
def token_info():
    global tokenizer
    if tokenizer is None:
        return jsonify({"error": "Tokenizer not trained"}), 400

    data = request.get_json()
    tokens = data.get("tokens", [])

    if not tokens:
        return jsonify({"error": "No tokens provided"}), 400

    token_info = [tokenizer.decode([token]) for token in tokens]
    return jsonify({"token_info": token_info})

if __name__ == '__main__':
    app.run(debug=True)