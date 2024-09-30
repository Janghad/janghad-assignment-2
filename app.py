from flask import Flask, render_template, request, jsonify
import random
from kMeans import kmeans_stepwise

app = Flask(__name__)

# Global variable to store the dataset
data_points = [[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(300)]

@app.route('/')
def index():
    return render_template('index.html', points=data_points, steps=[])

@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    global data_points
    data_points = [[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(500)]
    return jsonify({"message": "New dataset generated successfully", "points": data_points})

@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    data = request.get_json()  # Expecting JSON from fetch request
    k = int(data['k'])
    init_method = data.get('init_method', 'random')
    manual_centroids = data.get('manual_centroids', None)

    if manual_centroids and init_method == "manual":
        steps = kmeans_stepwise(data_points, k, init_method, manual_centroids=manual_centroids)
    else:
        steps = kmeans_stepwise(data_points, k, init_method)

    return jsonify({"steps": steps, "points": data_points})

if __name__ == '__main__':
    app.run(debug=True)
