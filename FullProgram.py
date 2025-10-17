import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from openai import OpenAI
from sklearn.preprocessing import RobustScaler
import joblib
import json
import os

class CholesterolAdvisor:
    def __init__(self):
        current_directory = os.getcwd()

        self.cholesterol_predictor = load_model(os.path.join(current_directory, 'models', 'cholesterol_model', 'cholesterol_predictor.h5'))
        self.food_recommender = load_model(os.path.join(current_directory, 'models', 'food_model', 'food_recommender.h5'))

        self.cholesterol_scaler = joblib.load(os.path.join(current_directory, 'models', 'cholesterol_model', 'cholesterol_scaler.joblib'))
        self.food_scaler = joblib.load(os.path.join(current_directory, 'models', 'food_model', 'food_scaler.joblib'))

        with open(os.path.join(current_directory, 'models', 'food_model', 'food_features.json'), 'r') as f:
            self.food_features = json.load(f)

        self.food_database = pd.read_csv("/content/processed_diet_data_corrected.csv", low_memory=False)

        self.food_database = self.food_database.dropna(subset=['food_name'])

    def predict_cholesterol_needs(self, hdl, ldl, total):
        features = {
            'HDL_Level': hdl,
            'LDL_Level': ldl,
            'Total_Cholesterol': total,
            'LDL_to_HDL_Ratio': ldl / (hdl + 1),
            'Cholesterol_Difference': total - ldl
        }


        features_df = pd.DataFrame([features])
        scaled_features = self.cholesterol_scaler.transform(features_df)

        predictions = self.cholesterol_predictor.predict(scaled_features)

        hdl_pred = np.argmax(predictions[0], axis=1)[0] - 1
        ldl_pred = np.argmax(predictions[1], axis=1)[0] - 1
        total_pred = np.argmax(predictions[2], axis=1)[0] - 1

        return {
            'HDL_needs': self._interpret_prediction(hdl_pred),
            'LDL_needs': self._interpret_prediction(ldl_pred),
            'Total_needs': self._interpret_prediction(total_pred)
        }

    def recommend_foods(self, cholesterol_needs):
        try:
            filtered_foods = self.food_database[
                ((self.food_database['HDL_impact'] == 1) if cholesterol_needs['HDL_needs'] == 'increase' else
                 (self.food_database['HDL_impact'] == -1) if cholesterol_needs['HDL_needs'] == 'decrease' else True) &
                ((self.food_database['LDL_impact'] == -1) if cholesterol_needs['LDL_needs'] == 'decrease' else
                 (self.food_database['LDL_impact'] == 1) if cholesterol_needs['LDL_needs'] == 'increase' else True) &
                ((self.food_database['Total_Cholesterol_impact'] == -1) if cholesterol_needs['Total_needs'] == 'decrease' else
                 (self.food_database['Total_Cholesterol_impact'] == 1) if cholesterol_needs['Total_needs'] == 'increase' else True)
            ]

            filtered_foods = filtered_foods[~filtered_foods['food_name'].str.contains('BAR|MARGARINE|SPREAD|PROCESSED', case=True, na=False)]

            return filtered_foods[['food_name', 'HDL_impact', 'LDL_impact', 'Total_Cholesterol_impact']].head(5)

        except Exception as e:
            print(f"Debug - Error in recommend_foods: {str(e)}")
            return pd.DataFrame([{"food_name": "Error getting food recommendations", "HDL_impact": 0, "LDL_impact": 0, "Total_Cholesterol_impact": 0}])

    def _interpret_prediction(self, pred):
        if pred == -1:
            return 'decrease'
        elif pred == 1:
            return 'increase'
        else:
            return 'maintain'

def validate_cholesterol_levels(hdl, ldl, total):
    """Validate cholesterol levels are within reasonable ranges"""
    if not (20 <= hdl <= 100):
        raise ValueError("HDL should be between 20 and 100 mg/dL")
    if not (40 <= ldl <= 300):
        raise ValueError("LDL should be between 40 and 300 mg/dL")
    if not (100 <= total <= 400):
        raise ValueError("Total Cholesterol should be between 100 and 400 mg/dL")
    if total < (hdl + ldl):
        raise ValueError("Total Cholesterol should be greater than or equal to HDL + LDL")

def main():
    advisor = CholesterolAdvisor()

    print("Welcome to the Cholesterol Management Advisor!")
    print("\nPlease enter your cholesterol levels (in mg/dL):")
    print("Healthy ranges for reference:")
    print("HDL: Above 40 (men) or 50 (women)")
    print("LDL: Below 100")
    print("Total: Below 200")

    try:
        hdl = float(input("\nHDL (Good cholesterol) level: "))
        ldl = float(input("LDL (Bad cholesterol) level: "))
        total = float(input("Total Cholesterol level: "))

        validate_cholesterol_levels(hdl, ldl, total)

        needs = advisor.predict_cholesterol_needs(hdl, ldl, total)

        print("\n Your Cholesterol Profile:")
        print(f"HDL: {hdl} mg/dL - {'Good' if hdl >= 50 else 'Needs Improvement'}")
        print(f"LDL: {ldl} mg/dL - {'Good' if ldl < 100 else 'Needs Improvement'}")
        print(f"Total: {total} mg/dL - {'Good' if total < 200 else 'Needs Improvement'}")

        print("\n Recommended Actions:")
        print(f"HDL (Good cholesterol): {needs['HDL_needs']}")
        print(f"LDL (Bad cholesterol): {needs['LDL_needs']}")
        print(f"Total Cholesterol: {needs['Total_needs']}")

        print("\n Recommended foods for your cholesterol profile:")

        recommended_foods = advisor.recommend_foods(needs)

        impact_map = {-1: "↓ decreases", 0: "→ neutral", 1: "↑ increases"}
        for i, row in recommended_foods.iterrows():
            print(f"{i+1}. {row['food_name']}")
            print(f"   - HDL impact: {impact_map.get(row['HDL_impact'], 'N/A')}")
            print(f"   - LDL impact: {impact_map.get(row['LDL_impact'], 'N/A')}")
            print(f"   - Total Cholesterol impact: {impact_map.get(row['Total_Cholesterol_impact'], 'N/A')}")

    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
