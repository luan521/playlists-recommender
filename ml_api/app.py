import pandas as pd
from fpgrowth_py import fpgrowth
import pickle
import os
import sys

# Define a class to implement a recommendation system based on FP-Growth
class Recommender:
    def __init__(self, minSupRatio, minConf):
        # Initialize the recommender with minimum support ratio and confidence
        self.minSupRatio = minSupRatio
        self.minConf = minConf
        
    def fit(self, songSetList):
        # Generate frequent itemsets and association rules using FP-Growth
        freqItemSet, rules = fpgrowth(songSetList, minSupRatio=self.minSupRatio, minConf=self.minConf)
        # Create a DataFrame from the rules
        df_rules = pd.DataFrame(rules, columns=['antecedent', 'consequent', 'score'])
        # Simplify antecedent and consequent by extracting the first item in each set
        df_rules['antecedent'] = df_rules['antecedent'].apply(lambda x: list(x)[0])
        df_rules['consequent'] = df_rules['consequent'].apply(lambda x: list(x)[0])
        # Store the resulting rules as a class attribute
        self.rules = df_rules
        
    def predict(self, songList, n=20):
        songList = [s.lower() for s in songList]
        # Filter rules where antecedents match the provided song list
        index_filter = self.rules['antecedent'].isin(songList)
        # Group the rules by consequent and sum their scores, then sort to find top recommendations
        response = list(self.rules[index_filter]
                        .groupby('consequent')['score']
                        .sum()
                        .sort_values(ascending=False)
                        .index[:n])
        return response
    
from flask import Flask, request, jsonify

# Run the Flask app on a specific port
if __name__ == "__main__":
    # Parse command-line arguments
    args = sys.argv[1:]
    host = args[0]
    port = args[1]
    model_folder = args[2]

    # Initialize Flask app
    app = Flask(__name__)

    # Load the recommendation model
    models = os.listdir(model_folder)
    models.sort()
    model_version = int(models[-1][7:].replace('.pkl', ''))
    with open(f"{model_folder}/model-v{model_version}.pkl", "rb") as file:
        app.model = pickle.load(file)

    # Define the recommendation route
    @app.route("/api/recommender", methods=["POST"])
    def recommend():
        try:
            # Parse the input JSON
            data = request.get_json(force=True)
            if "songs" not in data:
                return jsonify({"error": "Missing 'songs' field in request"}), 400
            
            liked_songs = data["songs"]
            
            # Generate recommendations using the model
            recommendations = app.model.predict(liked_songs)
            
            # Create the response
            response = {
                "songs": recommendations
            }
            return jsonify(response), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    app.run(host=host, port=port)