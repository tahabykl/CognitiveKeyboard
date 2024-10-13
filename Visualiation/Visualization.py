import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime
import fetch

import requests
import json


def parse_json():
    data = fetch.getJson()
    ages = [entry['age'] for entry in data]
    races = [entry['race'] for entry in data]
    genders = [entry['gender'] for entry in data]
    # names = [entry['name'] for entry in data]
    # uuids = [entry['uuid'] for entry in data]

    # Initialize lists to hold the averages
    backspace_count = []
    interKeystrokeInterval = []
    characterVariability = []
    specialCharacterUsage = []
    typingSpeed = []

# Loop through the data entries
    for entry in data:
        # Initialize temporary lists for the current entry's metrics
        curr_typingAccuracy = []
        curr_interKeystrokeInterval = []
        curr_characterVariability = []
        curr_specialCharacterUsage = []
        curr_typingSpeed = []
        
        for pattern in entry['typing_patterns']:
            # Append each metric to its respective list, using 0 as a default if the key doesn't exist
            curr_typingAccuracy.append(pattern.get('typingAccuracy', 0))
            curr_interKeystrokeInterval.append(pattern.get('interKeystrokeInterval', 0))
            curr_characterVariability.append(pattern.get('characterVariability', 0))
            curr_specialCharacterUsage.append(pattern.get('specialCharacterUsage', 0))
            curr_typingSpeed.append(pattern.get('typingSpeed', 0))

        # Calculate and append the averages for each metric
        backspace_count.append(sum(curr_typingAccuracy) / len(curr_typingAccuracy) if curr_typingAccuracy else 0)
        interKeystrokeInterval.append(sum(curr_interKeystrokeInterval) / len(curr_interKeystrokeInterval) if curr_interKeystrokeInterval else 0)
        characterVariability.append(sum(curr_characterVariability) / len(curr_characterVariability) if curr_characterVariability else 0)
        specialCharacterUsage.append(sum(curr_specialCharacterUsage) / len(curr_specialCharacterUsage) if curr_specialCharacterUsage else 0)
        typingSpeed.append(sum(curr_typingSpeed) / len(curr_typingSpeed) if curr_typingSpeed else 0)


    # print(json.dumps(data, indent=4))
    # print("ages: ", ages)
    # print("races: ", races)
    # print("genders: ", genders)
    # print("names: ", names)
    # print("uuids: ", uuids)
    # print("backspace_count: ", backspace_count)


    pd_data = pd.DataFrame({
    'age': ages,
    'race': races,
    'gender': genders,
    'backspace count': backspace_count,
    'Inter Keystroke Interval': interKeystrokeInterval,
    'Character Variability': characterVariability,
    'Special Character Usage': specialCharacterUsage,
    'Typing Speed': typingSpeed


    # 'contact number': ['123-456-7890', '234-567-8901', '345-678-9012', '456-789-0123', '567-890-1234', '678-901-2345'],
    # 'backspace_count': [5, 3, 8, 2, 7, 4],
    # 'char_counts': [{}, {}, {}, {}, {}, {}],
    # 'special_char_count': [1, 2, 0, 3, 1, 2],
    # 'typing_speed': [50, 45, 60, 55, 48, 52]
    
    })

    st.write(pd_data.iloc[:])


# Can siply connect to the pandas.
# Sample data to test the visualization

parse_json()


data = pd.DataFrame({
    'age': [25, 30, 22, 28, 35, 40],
    'race': ['Asian', 'Caucasian', 'Hispanic', 'African American', 'Caucasian', 'Asian'],
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'contact number': ['123-456-7890', '234-567-8901', '345-678-9012', '456-789-0123', '567-890-1234', '678-901-2345'],
    'timestamp': [1620000000, 1620003600, 1620007200, 1620010800, 1625125800, 1605125800],
    'backspace_count': [5, 3, 8, 2, 7, 4],
    'char_counts': [{}, {}, {}, {}, {}, {}],
    'special_char_count': [1, 2, 0, 3, 1, 2],
    'typing_speed': [50, 45, 60, 55, 48, 52]
})

# Normalize column names
data.columns = data.columns.str.strip().str.lower()

# Convert timestamp to datetime format for filtering
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

# Streamlit app
st.title("MongoDB Data Visualization and Filtering")

# Sidebar filters
st.sidebar.header("Filters")

# Age filter options
age_options = ["All", "3-18", "25-32", "32-50", "60+"]
age_filter = st.sidebar.selectbox("Select Age Range", options=age_options)

# Gender filter options
gender_options = ["All", "Male", "Female"]
gender_filter = st.sidebar.selectbox("Select Gender", options=gender_options)

# Race filter options
race_options = ["All"] + list(data['race'].unique())
race_filter = st.sidebar.selectbox("Select Race", options=race_options)

# Calculate min and max dates from the dataset
min_date = data['timestamp'].min().date()
max_date = data['timestamp'].max().date()

# Timestamp filter (start time and end time)
st.sidebar.write("Select Date Range")
start_date = st.sidebar.date_input("Start Date", value=min_date)
end_date = st.sidebar.date_input("End Date", value=max_date)

# Filtering the data based on the filters
filtered_data = data.copy()

# Apply age filter
if age_filter != "All":
    if age_filter == "3-18":
        filtered_data = filtered_data[(filtered_data['age'] >= 3) & (filtered_data['age'] <= 18)]
    elif age_filter == "25-32":
        filtered_data = filtered_data[(filtered_data['age'] >= 25) & (filtered_data['age'] <= 32)]
    elif age_filter == "32-50":
        filtered_data = filtered_data[(filtered_data['age'] >= 32) & (filtered_data['age'] <= 50)]
    elif age_filter == "60+":
        filtered_data = filtered_data[filtered_data['age'] >= 60]

# Apply gender filter
if gender_filter != "All":
    filtered_data = filtered_data[filtered_data['gender'] == gender_filter]

# Apply race filter
if race_filter != "All":
    filtered_data = filtered_data[filtered_data['race'] == race_filter]

# Apply timestamp filter
filtered_data = filtered_data[(filtered_data['timestamp'].dt.date >= start_date) & (filtered_data['timestamp'].dt.date <= end_date)]
st.write("### Filtered Data")
if filtered_data.empty:
    st.write("No data available for the selected filters.")
else:
    st.write(filtered_data.iloc[:, :5])  # Limit the displayed columns to 5

    # Clustering the data
    st.sidebar.header("Clustering")
    if len(filtered_data) > 1:
        num_clusters = st.sidebar.slider("Select Number of Clusters", 1, min(len(filtered_data), 5), 3)

        data_for_clustering = filtered_data[['age', 'backspace_count', 'special_char_count', 'typing_speed']]
        data_for_clustering = data_for_clustering.replace(0, np.nan).fillna(data_for_clustering.mean())

        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(data_for_clustering)
        filtered_data['Cluster'] = kmeans.labels_

        # Cluster labels using cluster_dict
        cluster_dict = {3: ["Parkinson's", "Healthy", "Tremor"]}  # Example cluster dictionary
        cluster_labels = {}
        for cluster_num in range(num_clusters):
            cluster_labels[cluster_num] = st.sidebar.text_input(f"Label for Cluster {cluster_num}", value=cluster_dict.get(num_clusters, [f"Cluster {cluster_num}"])[cluster_num])
        filtered_data['Cluster Label'] = filtered_data['Cluster'].map(cluster_labels)

        # Visualization of clusters in 2D and 3D
        st.write("### Data Visualizations")

        # Select dimensions for visualization
        st.sidebar.header("Visualization Dimensions")
        x_axis = st.sidebar.selectbox("Select X Axis", options=data_for_clustering.columns)
        y_axis = st.sidebar.selectbox("Select Y Axis", options=data_for_clustering.columns)
        z_axis = st.sidebar.selectbox("Select Z Axis (Optional for 3D)", options=[None] + list(data_for_clustering.columns))

        # 2D Scatter Plot
        st.write("#### 2D Scatter Plot")
        fig_2d = px.scatter(filtered_data, x=x_axis, y=y_axis, color='Cluster Label', title=f'2D Scatter Plot: {x_axis} vs {y_axis}', hover_name='contact number')
        st.plotly_chart(fig_2d)

        # 3D Scatter Plot (if Z Axis is selected)
        if z_axis and z_axis != 'None':
            st.write("#### 3D Scatter Plot")
            fig_3d = px.scatter_3d(filtered_data, x=x_axis, y=y_axis, z=z_axis, color='Cluster Label', title=f'3D Scatter Plot: {x_axis} vs {y_axis} vs {z_axis}', hover_name='contact number')
            st.plotly_chart(fig_3d)

        # Correlation Heatmap
        st.write("#### Correlation Heatmap")
        corr_matrix = data_for_clustering.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Parallel Coordinates Plot
        st.write("#### Parallel Coordinates Plot")
        fig_parallel = px.parallel_coordinates(filtered_data, dimensions=data_for_clustering.columns, color='Cluster', title='Parallel Coordinates Plot')
        st.plotly_chart(fig_parallel)

        # Interactive Summary Statistics
        st.write("#### Interactive Summary Statistics")
        numeric_cols = ['age', 'backspace_count', 'special_char_count', 'typing_speed']
        cluster_summary = filtered_data.groupby('Cluster Label')[numeric_cols].agg(['mean', 'median', 'std'])
        st.write(cluster_summary)
    else:
        st.write("Not enough data points for clustering.")

st.write("### Visualization Complete")


