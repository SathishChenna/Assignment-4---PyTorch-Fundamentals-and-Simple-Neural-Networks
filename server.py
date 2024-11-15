from flask import Flask, render_template, jsonify, request
import json
from pathlib import Path

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_training_data')
def get_training_data():
    try:
        with open('static/training_data.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({
            'model1': {'losses': [], 'train_accuracies': [], 'test_accuracies': []},
            'model2': {'losses': [], 'train_accuracies': [], 'test_accuracies': []}
        })

@app.route('/start_training', methods=['POST'])
def start_training():
    params = request.json
    # Save parameters to be used by training script
    with open('static/training_params.json', 'w') as f:
        json.dump(params, f)
    # Create a flag file to indicate training should start
    Path('static/start_training').touch()
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True) 