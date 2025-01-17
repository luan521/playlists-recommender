import pandas as pd
from fpgrowth_py import fpgrowth
import pickle
import os
import logging
import re

def clean_song_title(title):
    # Remove text in parentheses (e.g., "(feat. Ja Rule)")
    title = re.sub(r"\(.*?\)", "", title)
    # Remove text after hyphen (e.g., "- Main", "- Remastered 2011")
    title = re.sub(r"- .*", "", title)
    # Trim any extra whitespace
    return title.strip().lower()

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Set the log format
    handlers=[
        logging.StreamHandler()  # Print logs to the console
    ]
)

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
    
# Main script to generate the recommendation rules
if __name__ == '__main__':
    logging.info("INFO: Model fit started")
    minSupRatio = float(os.getenv("minSupRatio"))
    minConf = float(os.getenv("minConf"))
    model_folder = os.getenv("model_folder")
    dataset_path = os.getenv("dataset_path")
    logging.info("SUCCESS: Load env variables")
    
    # Read the dataset and group songs into sets by playlist ID
    df = pd.read_csv(dataset_path)
    df['track_name']=df['track_name'].apply(clean_song_title)
    songSetList = df.groupby('pid')['track_name'].apply(list).values
    logging.info("SUCCESS: Data load and transform")
    
    # Initialize and train the recommender model
    model = Recommender(minSupRatio=minSupRatio, minConf=minConf)
    model.fit(songSetList)
    logging.info("SUCCESS: Train the recommender model")
    
    models = os.listdir(model_folder)
    models.sort()
    model_version = int(models[-1][7:].replace('.pkl', '')) if len(models) > 0 else 0
    model_version+=1
    # Save the trained model as a pickle file for future use
    with open(f"{model_folder}/model-v{model_version}.pkl", "wb") as file:
        pickle.dump(model, file)
    logging.info(f"SUCCESS: Save the model in {model_folder}/model-v{model_version}.pkl")