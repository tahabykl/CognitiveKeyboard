import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import logging

# MongoDB connection setup
mongo_url = "mongodb+srv://test:5sPpV890LGojFLbe@newuser.q1nbs.mongodb.net/?retryWrites=true&w=majority&appName=newUser"
client = MongoClient(mongo_url, server_api=ServerApi('1'))

if not mongo_url:
    logging.error("MongoDB connection URL not set. Please set the 'MONGO_URL' environment variable.")
    raise Exception("MongoDB connection URL not set.")

client = MongoClient(mongo_url, server_api=ServerApi('1'))

db = client['newUser']
clusters_collection = db['clusters']

# Fetch data from MongoDB
clusters_data = clusters_collection.find()

data_list = []
for cluster in clusters_data:
    for cluster_data in cluster['clusters']:
        for data_point in cluster_data['data_points']:
            data_list.append({
                "Age": data_point['age'],
                "Race": data_point['race'],
                "Gender": data_point['gender'],
                "Typing Speed": data_point['typingSpeed'],
                "Interkey Interval": data_point['interKeystrokeInterval'],
                "Typing Accuracy": data_point['typingAccuracy'],
                "Character Variability": data_point['characterVariability'],
                "Special Character Usage": data_point['specialCharacterUsage'],
                "Cluster Label": data_point['Cluster Label']
            })

# Create pandas DataFrame
df = pd.DataFrame(data_list)

# Convert numerical columns to appropriate types
df['Age'] = pd.to_numeric(df['Age'])
df['Typing Speed'] = pd.to_numeric(df['Typing Speed'])
df['Interkey Interval'] = pd.to_numeric(df['Interkey Interval'])
df['Typing Accuracy'] = pd.to_numeric(df['Typing Accuracy'])
df['Character Variability'] = pd.to_numeric(df['Character Variability'])
df['Special Character Usage'] = pd.to_numeric(df['Special Character Usage'])

# Preprocess data
race_encoder = LabelEncoder()
gender_encoder = LabelEncoder()
cluster_label_encoder = LabelEncoder()
df['Race'] = race_encoder.fit_transform(df['Race'])
df['Gender'] = gender_encoder.fit_transform(df['Gender'])
df['Cluster Label'] = cluster_label_encoder.fit_transform(df['Cluster Label'])

X = df[['Age', 'Race', 'Gender', 'Typing Speed', 'Interkey Interval', 'Typing Accuracy', 'Character Variability', 'Special Character Usage']]
y = df['Cluster Label']

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

def main():
    st.title("Patient Data Entry Screen")
    st.write("Please enter the parameters below to evaluate the patient's typing data.")

    # Input fields for patient parameters
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    race = st.selectbox("Race", ["Caucasian", "African American", "Asian", "Hispanic", "Other"])
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    typing_speed = st.number_input("Typing Speed (words per minute)", min_value=0.0, step=1.0)
    interkey_interval = st.number_input("Interkey Strip Interval (ms)", min_value=0.0, step=1.0)
    typing_accuracy = st.slider("Typing Accuracy (%)", min_value=0, max_value=100, value=100)
    char_variability = st.number_input("Character Variability", min_value=0.0, step=0.1)
    special_char_usage = st.number_input("Special Character Usage (%)", min_value=0.0, max_value=100.0, step=1.0)

    # Button to submit data
    if st.button("Submit Data"):
        st.write("### Data Entered:")
        st.write(f"Age: {age}")
        st.write(f"Race: {race}")
        st.write(f"Gender: {gender}")
        st.write(f"Typing Speed: {typing_speed} wpm")
        st.write(f"Interkey Strip Interval: {interkey_interval} ms")
        st.write(f"Typing Accuracy: {typing_accuracy} %")
        st.write(f"Character Variability: {char_variability}")
        st.write(f"Special Character Usage: {special_char_usage} %")

        # Encode input data
        input_data = pd.DataFrame({
            "Age": [age],
            "Race": race_encoder.transform([race]),
            "Gender": gender_encoder.transform([gender]),
            "Typing Speed": [typing_speed],
            "Interkey Interval": [interkey_interval],
            "Typing Accuracy": [typing_accuracy],
            "Character Variability": [char_variability],
            "Special Character Usage": [special_char_usage]
        })

        # Make prediction
        prediction = knn.predict(input_data)
        predicted_label = cluster_label_encoder.inverse_transform(prediction)[0]

        # Display diagnosis
        st.write("### Diagnosis:")
        st.write(f"The predicted cluster label for the patient is: {predicted_label}")

if __name__ == "__main__":
    main()