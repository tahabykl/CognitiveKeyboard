from flask import Flask, request, jsonify
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import uuid
import decimal
from datetime import datetime, timedelta
import os
from bson import ObjectId
import bson
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(
    filename='app.log',  # Log file name
    level=logging.INFO,  # Log level
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)


mongo_url = "mongodb+srv://test:5sPpV890LGojFLbe@newuser.q1nbs.mongodb.net/?retryWrites=true&w=majority&appName=newUser&ssl=true&sslVerifyCertificate=false"
client = MongoClient(mongo_url, server_api=ServerApi('1'))

if not mongo_url:
    logging.error("MongoDB connection URL not set. Please set the 'MONGO_URL' environment variable.")
    raise Exception("MongoDB connection URL not set.")

client = MongoClient(mongo_url, server_api=ServerApi('1'))

db = client['newUser']  # Replace with your database name
users_collection = db['users']  # Replace with your collection name
typing_collection = db['typing_data']
clusters_collection = db['clusters']
feedback_collection = db['feedback']

@app.route('/test', methods=['POST'])
def test():
    return 'hello'

def make_serializable(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    # Add more conversions if necessary
    return obj


@app.route('/create_uuid', methods=['POST'])
def register():
    try:
        user_uuid = str(uuid.uuid4())
        return user_uuid
    except Exception as e:
        logging.error(f"Error in register: {str(e)}")
        return "hello"


@app.route('/p', methods=['POST'])
def post_clusters():
    try:
        data = request.json
        print("Incoming data:", data)  # Debugging: print incoming data

        # Ensure the incoming data is not None or empty
        if not data:
            return jsonify({'message': 'No data provided'}), 400

        # Perform the update operation
        update_result = clusters_collection.update_one(
            {"_id": bson.ObjectId("670aaddb32d5e1c0e26c96c7")},
            {"$set": data}
        )

        # Check if the document was modified
        if update_result.modified_count > 0:
            return jsonify({'message': 'Cluster document updated successfully'}), 200
        else:
            return jsonify({'message': 'No changes made to the cluster document'}), 200
    except Exception as e:
        print("Error:", str(e))  # Debugging: print the exception
        return jsonify({'error': str(e)}), 500


# Add a new user to the database with demographic info and a unique id
@app.route('/new-user', methods=['POST'])
def new_user():
    try:
        logging.debug("Entered new_user function")
        data = request.json
        logging.debug(f"Request data: {data}")
        user_uuid = str(uuid.uuid4())
        logging.debug(f"Generated UUID: {user_uuid}")

        user_data = {
            "uuid": user_uuid,
            "name": data['name'],
            "age": data['age'],
            "race": data['race'],
            "gender": data['gender'],
            "contact": data['contact'],
            "conditions": data['conditions']
        }

        logging.debug("Created user data dictionary")
        # Insert the data into the MongoDB collection
        users_collection.insert_one(user_data)
        logging.info(f"Inserted new user into collection with UUID: {user_uuid}")
        # Return a success message with the inserted document's id
        return jsonify({'message': 'Entry added successfully', 'uuid': user_uuid}), 201

    except Exception as e:
        logging.error(f"Error in new_user: {str(e)}")
        return jsonify({'error': 'An error occurred'}), 500

# Either create a list of typing data for a user or update their existing list based on uuid
@app.route('/typing-data/<uuid>', methods=['POST'])
def new_data(uuid):
    try:
        collection = db[f'day{datetime.now().strftime("%Y-%m-%d")}']
        data = request.json
        typing_data = {
            "typingSpeed": data['typingSpeed'],
            "interKeystrokeInterval": data['interKeystrokeInterval'],
            "typingAccuracy": data['typingAccuracy'],
            "characterVariability": data['characterVariability'],
            "specialCharacterUsage": data['specialCharacterUsage']
        }
        existing_entry = collection.find_one({"uuid": uuid})
        if existing_entry:
            collection.update_one(
                {"uuid": uuid},
                {"$push": {"typing_data_list": typing_data}}
            )
            logging.debug(f"Updated typing data for UUID: {uuid}")
        else:
            collection.insert_one({
                "uuid": uuid,
                "typing_data_list": [typing_data]}
            )
            logging.debug(f"Inserted new typing data for UUID: {uuid}")
        return jsonify({'message': 'Typing data added successfully', 'uuid': uuid}), 201
    except Exception as e:
        logging.error(f"Error in new_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/test-typing-data/<uuid>/<date>', methods=['POST'])
def test_typing(uuid, date):
    try:
        collection = db[f'day{date}']
        data = request.json
        typing_data = {
            "typingSpeed": data['typingSpeed'],
            "interKeystrokeInterval": data['interKeystrokeInterval'],
            "typingAccuracy": data['typingAccuracy'],
            "characterVariability": data['characterVariability'],
            "specialCharacterUsage": data['specialCharacterUsage']
        }
        existing_entry = collection.find_one({"uuid": uuid})
        if existing_entry:
            collection.update_one(
                {"uuid": uuid},
                {"$push": {"typing_data_list": typing_data}}
            )
            logging.debug(f"Updated typing data for UUID: {uuid} on date: {date}")
        else:
            collection.insert_one({
                "uuid": uuid,
                "typing_data_list": [typing_data]}
            )
            logging.debug(f"Inserted new typing data for UUID: {uuid} on date: {date}")
        return jsonify({'message': 'Typing data added successfully', 'uuid': uuid}), 201
    except Exception as e:
        logging.error(f"Error in test_typing: {str(e)}")
        return jsonify({'error': str(e)}), 500


def make_serializable(value):
    """
    Convert MongoDB BSON types to JSON-serializable types.
    """
    if isinstance(value, ObjectId):
        return str(value)
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, list):
        return [make_serializable(v) for v in value]
    elif isinstance(value, dict):
        return {k: make_serializable(v) for k, v in value.items()}
    else:
        return value


@app.route('/metrics/<start_date>/<end_date>/<uuid>', methods=['GET'])
def metrics(start_date, end_date, uuid):
    try:
        # Parse the start and end dates
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Initialize an empty list to collect typing data
        typing_patterns = []

        current_date = start_date_dt
        while current_date <= end_date_dt:
            collection_name = f'day{current_date.strftime("%Y-%m-%d")}'
            collection = db[collection_name]

            # Fetch typing data for the user on the current date
            typing_data = collection.find_one({"uuid": uuid})
            logging.debug(f"Typing data for user {uuid} on {current_date.strftime('%Y-%m-%d')}: {typing_data}")

            if typing_data:
                # Retrieve typing data list
                typing_data_list = typing_data.get('typing_data_list') or typing_data.get('typing_data')
                if typing_data_list:
                    # Make the typing data serializable
                    typing_data_list = [{k: make_serializable(v) for k, v in item.items()} for item in typing_data_list]
                    logging.info(f"Retrieved typing_data_list: {typing_data_list}")

                    # Extend typing_patterns with all items from typing_data_list
                    typing_patterns.extend(typing_data_list)
                else:
                    logging.debug(f"No typing data list found for user {uuid} on {current_date.strftime('%Y-%m-%d')}")
            else:
                logging.debug(f"No typing data found for user {uuid} on {current_date.strftime('%Y-%m-%d')}")

            current_date += timedelta(days=1)

        # Return the typing_patterns list
        return jsonify(typing_patterns), 200

    except Exception as e:
        logging.error(f"Error in metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/all-data/<start_date>/<end_date>', methods=['GET'])
def all_entries(start_date, end_date):
    try:
        # Parse the start and end dates
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Fetch all users
        users = list(users_collection.find())
        user_data_map = {}

        current_date = start_date_dt
        while current_date <= end_date_dt:
            collection_name = f'day{current_date.strftime("%Y-%m-%d")}'
            collection = db[collection_name]

            for user in users:
                user_uuid = user['uuid']
                if user_uuid not in user_data_map:
                    user_data_map[user_uuid] = {
                        "uuid": user_uuid,
                        "name": user.get('name'),
                        "age": user.get('age'),
                        "race": user.get('race'),
                        "gender": user.get('gender'),
                        "conditions": user.get('conditions'),
                        "contact": user.get('contact'),
                        "typing_patterns": []
                    }
                # Fetch typing data for the user on the current date
                typing_data = collection.find_one({"uuid": user_uuid})
                logging.debug(f"Typing data for user {user_uuid} on {current_date.strftime('%Y-%m-%d')}: {typing_data}")

                if typing_data:
                    # Retrieve typing data list
                    typing_data_list = typing_data.get('typing_data_list') or typing_data.get('typing_data')
                    if typing_data_list:
                        # Make the typing data serializable
                        typing_data_list = [{k: make_serializable(v) for k, v in item.items()} for item in
                                            typing_data_list]
                        logging.info(f"Retrieved typing_data_list: {typing_data_list}")

                        # Extend typing_patterns with all items from typing_data_list
                        user_data_map[user_uuid]["typing_patterns"].extend(typing_data_list)
                    else:
                        logging.debug(
                            f"No typing data list found for user {user_uuid} on {current_date.strftime('%Y-%m-%d')}")
                else:
                    logging.debug(f"No typing data found for user {user_uuid} on {current_date.strftime('%Y-%m-%d')}")

            current_date += timedelta(days=1)
            logging.info(f"Processed data for date {current_date.strftime('%Y-%m-%d')}")

        # Ensure the entire user_data_map is serializable
        serializable_user_data_map = {k: make_serializable(v) for k, v in user_data_map.items()}

        return jsonify(serializable_user_data_map), 200

    except Exception as e:
        logging.error(f"Error in all_entries: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/feedback/<uuid>/<label>', methods=['POST'])
def post_feedback(uuid, label):
    try:
        data = request.json
        feedback_data = {
            "uuid": uuid,
            "age": data['age'],
            "race": data['race'],
            "gender": data['gender'],
            "accuracy": data['accuracy'],
            "speed": data['speed'],
            "interKeystrokeInterval": data['interKeystrokeInterval'],
            "label":label
        }
        feedback_collection.insert_one(feedback_data)
        return jsonify({'message': 'Feedback added successfully', 'uuid': uuid}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/feedback/<uuid>', methods=['GET'])
def get_feedback(uuid):
    try:
        feedback_entry = feedback_collection.find_one({"uuid": uuid})
        if feedback_entry:
            feedback_entry.pop('_id', None)
            return jsonify(feedback_entry), 200
        else:
            logging.warning(f"Feedback entry not found for UUID: {uuid}")
            return jsonify({'error': 'Feedback entry not found'}), 404
    except Exception as e:
        logging.error(f"Error in get_feedback: {str(e)}")
        return jsonify({'error': 'An error occurred'}), 500

@app.route('/fetchClusters', methods=['GET'])
def fetch_clusters():
    try:
        cluster_document = clusters_collection.find_one({"_id": bson.ObjectId("670aaddb32d5e1c0e26c96c7")})
        if cluster_document:
            # Convert ObjectId to string
            cluster_document['_id'] = str(cluster_document['_id'])
            return jsonify(cluster_document), 200
        else:
            return jsonify({"message": "Cluster document not found"}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)
