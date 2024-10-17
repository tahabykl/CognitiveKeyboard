import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import requests
import logging

# ===========================
# Configure Logging
# ===========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# Streamlit App Initialization
# ===========================
st.title("MongoDB Data Visualization and Filtering")

# ===========================
# Define API Base URLs
# ===========================
base_url = "https://7bc3-67-134-204-47.ngrok-free.app"
metrics_base_url = "https://7bc3-67-134-204-47.ngrok-free.app"

# ===========================
# Sidebar Filters
# ===========================
st.sidebar.header("Filters")

# ---------------------------
# Date Range Selection
# ---------------------------
st.sidebar.header("Select Date Range")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=7))
end_date = st.sidebar.date_input("End Date", value=datetime.now())

if start_date > end_date:
    st.error("Start Date must be before End Date.")
    st.stop()

# ---------------------------
# Fetch and Process Data
# ---------------------------
try:
    response = requests.get(
        f"{base_url}/all-data/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    )
    if response.status_code == 200:
        data_json = response.json()
        if not data_json:
            st.warning("No data available for the selected date range.")
            st.stop()
    else:
        st.error(f"Failed to fetch data from the server.\nStatus Code: {response.status_code}\nResponse: {response.text}")
        logger.error(f"Failed to fetch data from the server.\nStatus Code: {response.status_code}\nResponse: {response.text}")
        st.stop()
except Exception as e:
    st.error(f"An error occurred while fetching data: {e}")
    logger.exception("Exception occurred during data fetching")
    st.stop()

users_list = []
excluded_users = 0  # Counter for excluded users

for user_uuid, user_data in data_json.items():
    uuid = user_data.get('uuid')
    age = user_data.get('age')
    race = user_data.get('race')
    gender = user_data.get('gender')

    # Fetch typing patterns for the user from the metrics endpoint
    try:
        metrics_response = requests.get(
            f"{metrics_base_url}/metrics/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}/{uuid}"
        )
        if metrics_response.status_code == 200:
            typing_patterns = metrics_response.json()
            if not typing_patterns:
                logger.warning(f"No typing data available for user {uuid} for the selected date range.")
                typing_patterns = []
        else:
            logger.error(f"Failed to fetch typing data for user {uuid}.\nStatus Code: {metrics_response.status_code}\nResponse: {metrics_response.text}")
            typing_patterns = []
    except Exception as e:
        logger.error(f"An error occurred while fetching typing data for user {uuid}: {e}")
        typing_patterns = []

    # Initialize accumulators
    total_typing_speed = 0
    total_inter_keystroke_interval = 0
    total_typing_accuracy = 0
    total_character_variability = 0
    total_special_character_usage = 0
    count = 0

    # Process typing patterns
    for pattern in typing_patterns:
        # Extract typing metrics, assuming these keys exist
        typing_speed = pattern.get('typingSpeed', np.nan)
        inter_keystroke_interval = pattern.get('interKeystrokeInterval', np.nan)
        typing_accuracy = pattern.get('typingAccuracy', np.nan)
        character_variability = pattern.get('characterVariability', np.nan)
        special_character_usage = pattern.get('specialCharacterUsage', np.nan)

        # Accumulate metrics if they are valid numbers
        if not np.isnan(typing_speed):
            total_typing_speed += typing_speed
        if not np.isnan(inter_keystroke_interval):
            total_inter_keystroke_interval += inter_keystroke_interval
        if not np.isnan(typing_accuracy):
            total_typing_accuracy += typing_accuracy
        if not np.isnan(character_variability):
            total_character_variability += character_variability
        if not np.isnan(special_character_usage):
            total_special_character_usage += special_character_usage
        count += 1

    # Calculate average metrics
    if count > 0:
        avg_typing_speed = total_typing_speed / count
        avg_inter_keystroke_interval = total_inter_keystroke_interval / count
        avg_typing_accuracy = total_typing_accuracy / count
        avg_character_variability = total_character_variability / count
        avg_special_character_usage = total_special_character_usage / count
    else:
        # Assign NaN to typing metrics if no data is available
        avg_typing_speed = np.nan
        avg_inter_keystroke_interval = np.nan
        avg_typing_accuracy = np.nan
        avg_character_variability = np.nan
        avg_special_character_usage = np.nan

    # Exclude users with any NaN typing metrics
    if not any(pd.isna([
        avg_typing_speed,
        avg_inter_keystroke_interval,
        avg_typing_accuracy,
        avg_character_variability,
        avg_special_character_usage
    ])):
        user_dict = {
            'uuid': uuid,
            'age': age,
            'race': race,
            'gender': gender,
            'typingSpeed': avg_typing_speed,
            'interKeystrokeInterval': avg_inter_keystroke_interval,
            'typingAccuracy': avg_typing_accuracy,
            'characterVariability': avg_character_variability,
            'specialCharacterUsage': avg_special_character_usage
        }
        users_list.append(user_dict)
    else:
        excluded_users += 1
        logger.warning(f"User {uuid} has incomplete typing metrics and will be excluded.")

# Create DataFrame from the processed users
data = pd.DataFrame(users_list)

if data.empty:
    st.warning("No data available after processing.")
    st.stop()

# Display summary of excluded users
if excluded_users > 0:
    st.info(f"{excluded_users} user(s) were excluded due to incomplete typing metrics.")

# ===========================
# Demographic Filters
# ===========================
st.sidebar.header("Demographic Filters")

# Age filter options
age_options = ["All", "3-18", "25-32", "32-50", "60+"]
age_filter = st.sidebar.selectbox("Select Age Range", options=age_options)

# Gender filter options
gender_options = ["All"] + sorted(data['gender'].dropna().unique().tolist())
gender_filter = st.sidebar.selectbox("Select Gender", options=gender_options)

# Race filter options
race_options = ["All"] + sorted(data['race'].dropna().unique().tolist())
race_filter = st.sidebar.selectbox("Select Race", options=race_options)

# ===========================
# Apply Demographic Filters
# ===========================
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

# Display filtered data
st.write("### Filtered Data")
if filtered_data.empty:
    st.write("No data available for the selected filters.")
    st.stop()
else:
    st.write(filtered_data)

# ===========================
# Clustering and Visualization
# ===========================
st.sidebar.header("Clustering")

def send_feedback(data_points):
    for point in data_points:
        uuid = point['uuid']
        label = point['Cluster Label']
        payload = {
            "age": point['age'],
            "accuracy": point['typingAccuracy'],
            "speed": point['typingSpeed'],
            "interKeystrokeInterval": point['interKeystrokeInterval']
        }

        url = f"{base_url}/feedback/{uuid}/{label}"
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 201:
                print(f"Feedback for UUID {uuid} added successfully.")
            else:
                print(f"Failed to add feedback for UUID {uuid}. Status code: {response.status_code}, Error: {response.text}")
        except Exception as e:
            print(f"Error occurred while sending feedback for UUID {uuid}: {str(e)}")

if len(filtered_data) > 1:
    num_clusters = st.sidebar.slider(
        "Select Number of Clusters",
        1,
        min(len(filtered_data), 20),
        2,
        key='num_clusters'
    )

    # Prepare data for clustering
    # Selecting relevant columns for clustering
    clustering_columns = ['age', 'interKeystrokeInterval', 'specialCharacterUsage', 'typingSpeed']
    data_for_clustering = filtered_data[clustering_columns]

    # Replace 0 with NaN and fill with column means
    data_for_clustering = data_for_clustering.replace(0, np.nan).fillna(data_for_clustering.mean())
    print("*" * 100)
    print(data_for_clustering)
    print("*"*100)

    # Initialize KMeans with a fixed random state for reproducibility
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(data_for_clustering)


    filtered_data['Cluster'] = kmeans.labels_.astype(int)

    num_clusters_str = str(num_clusters)

    # Initialize cluster_labels in session state with default labels like "Category 1"
    if 'cluster_labels' not in st.session_state or st.session_state.get('prev_num_clusters') != num_clusters:
        st.session_state['cluster_labels'] = {}
        st.session_state['prev_num_clusters'] = num_clusters

        for cluster_num in range(num_clusters):
            st.session_state['cluster_labels'][cluster_num] = f'Category {cluster_num + 1}'

    cluster_labels = st.session_state['cluster_labels']

    # Map cluster labels to data before text inputs
    filtered_data['Cluster'] = filtered_data['Cluster'].astype(int)
    cluster_labels_int_keys = {int(k): v for k, v in cluster_labels.items()}
    filtered_data['Cluster Label'] = filtered_data['Cluster'].map(cluster_labels_int_keys)

    # Create text inputs for cluster labels
    cluster_labels_updated = {}
    for cluster_num in range(num_clusters):
        key = f'cluster_label_{cluster_num}_clusters_{num_clusters}'
        default_value = st.session_state['cluster_labels'][cluster_num]
        label = st.sidebar.text_input(
            f"Label for Cluster {cluster_num + 1}",
            value=default_value,
            key=key
        )
        cluster_labels_updated[cluster_num] = label

    # Update cluster labels in session state
    st.session_state['cluster_labels'] = cluster_labels_updated

    # Map updated cluster labels to data
    cluster_labels_int_keys = {int(k): v for k, v in cluster_labels_updated.items()}
    filtered_data['Cluster Label'] = filtered_data['Cluster'].map(cluster_labels_int_keys)

    # ===========================
    # Data Visualizations
    # ===========================
    st.write("### Data Visualizations")

    # Select dimensions for visualization
    st.sidebar.header("Visualization Dimensions")
    visualization_columns = data_for_clustering.columns.tolist()
    x_axis = st.sidebar.selectbox("Select X Axis", options=visualization_columns, key='x_axis')
    y_axis = st.sidebar.selectbox("Select Y Axis", options=visualization_columns, key='y_axis')
    z_axis = st.sidebar.selectbox(
        "Select Z Axis (Optional for 3D)",
        options=[None] + visualization_columns,
        key='z_axis'
    )

    # 2D Scatter Plot
    st.write("#### 2D Scatter Plot")
    fig_2d = px.scatter(
        filtered_data,
        x=x_axis,
        y=y_axis,
        color='Cluster Label',
        title=f'2D Scatter Plot: {x_axis} vs {y_axis}',
        hover_name='uuid'
    )
    st.plotly_chart(fig_2d)

    # 3D Scatter Plot (if Z Axis is selected)
    if z_axis and z_axis != 'None':
        st.write("#### 3D Scatter Plot")
        fig_3d = px.scatter_3d(
            filtered_data,
            x=x_axis,
            y=y_axis,
            z=z_axis,
            color='Cluster Label',
            title=f'3D Scatter Plot: {x_axis} vs {y_axis} vs {z_axis}',
            hover_name='uuid'
        )
        st.plotly_chart(fig_3d)

    # Correlation Heatmap
    st.write("#### Correlation Heatmap")
    corr_matrix = data_for_clustering.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Parallel Coordinates Plot
    st.write("#### Parallel Coordinates Plot")
    fig_parallel = px.parallel_coordinates(
        filtered_data,
        dimensions=visualization_columns,
        color='Cluster',
        title='Parallel Coordinates Plot',
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=filtered_data['Cluster'].mean()
    )
    st.plotly_chart(fig_parallel)

    # Interactive Summary Statistics
    st.write("#### Interactive Summary Statistics")
    numeric_cols = clustering_columns
    cluster_summary = filtered_data.groupby('Cluster Label')[numeric_cols].agg(['mean', 'median', 'std'])
    st.write(cluster_summary)

    # ===========================
    # Save and Insert Clusters
    # ===========================
    if st.button("Save and Insert Clusters"):
        # Update cluster_dict in session_state
        labels_list = [st.session_state['cluster_labels'][i] for i in range(num_clusters)]
        st.session_state['cluster_dict'] = st.session_state.get('cluster_dict', {})
        st.session_state['cluster_dict'][num_clusters_str] = labels_list

        # Prepare cluster data to send
        cluster_data_list = []
        for cluster_num in range(num_clusters):
            cluster_label = st.session_state['cluster_labels'][cluster_num]
            cluster_points = filtered_data[filtered_data['Cluster'] == cluster_num]
            # Convert cluster_points to a list of dictionaries
            cluster_points_dict = cluster_points.to_dict(orient='records')
            # Prepare cluster info
            cluster_info = {
                'cluster_num': cluster_num,
                'cluster_label': cluster_label,
                'data_points': cluster_points_dict
            }
            print(cluster_info)
            send_feedback(cluster_info["data_points"])
            cluster_data_list.append(cluster_info)

        # Prepare the payload to send
        cluster_save_data = {
            "cluster_dict": st.session_state['cluster_dict'],
            "clusters": cluster_data_list
        }
        print(cluster_save_data)

        # Display the payload for debugging (optional)
        # st.write("### Payload to be sent to the server")
        # st.json(cluster_save_data)

        # Send the data to the server with enhanced error reporting
        try:
            post_response = requests.post(f"{base_url}/p", json=st.session_state['cluster_dict'])
            if post_response.status_code == 200:
                st.success("Cluster labels and data successfully saved.")
            else:
                # Display detailed error information
                st.error(
                    f"Failed to save cluster labels and data.\n"
                    f"Status Code: {post_response.status_code}\n"
                    f"Response: {post_response.text}"
                )
                logger.error(
                    f"Failed to save cluster labels and data.\n"
                    f"Status Code: {post_response.status_code}\n"
                    f"Response: {post_response.text}"
                )
        except Exception as e:
            st.error(f"An exception occurred while saving clusters: {e}")
            logger.exception("Exception occurred during POST request to /postClusters")

else:
    st.write("Not enough data points for clustering.")

st.write("### Visualization Complete")