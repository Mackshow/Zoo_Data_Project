from flask import Flask, render_template, request
from pymongo import MongoClient
import pandas as pd
import joblib
from pycaret.classification import setup, predict_model
import numpy as np
import logging

# Out all requests and results into a log file
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
tuned_et_model = joblib.load(r'C:\Users\macks\OneDrive\Documents\Project 4\Test 1\tuned_svm_model.pkl')  

# Define route for homepage
@app.route('/')
@app.route('/index')
def index():
    # This function will be used to serve the homepage
    return render_template('index.html')

 # Define route for form submission   
@app.route('/submit', methods=['POST'])
def get_user_input():
    # This function will be used to handle form submission
    # It collects user input, makes predictions using the trained model, and returns the prediction
    animal_name = request.form['animal_name']
    hair = request.form['hair']
    feathers = request.form['feathers']
    eggs = request.form['eggs']
    milk = request.form['milk']
    airborne = request.form['airborne']
    aquatic = request.form['aquatic']
    predator = request.form['predator']
    toothed = request.form['toothed']
    backbone = request.form['backbone']
    air_breather = request.form['air_breather']
    water_breather = request.form['water_breather']
    venomous = request.form['venomous']
    fins = request.form['fins']
    tail = request.form['tail']
    legs = request.form['legs']

    # Store the user input in a dictionary
    animal_info = {
        "animal_name": animal_name,
        "hair": hair.lower() == 'yes',
        "feathers": feathers.lower() == 'yes',
        "eggs": eggs.lower() == 'yes',
        "milk": milk.lower() == 'yes',
        "airborne": airborne.lower() == 'yes',
        "aquatic": aquatic.lower() == 'yes',
        "predator": predator.lower() == 'yes',
        "toothed": toothed.lower() == 'yes',
        "backbone": backbone.lower() == 'yes',
        "air_breather": air_breather.lower() == 'yes',
        "water_breather": water_breather.lower() == 'yes',
        "venomous": venomous.lower() == 'yes',
        "fins": fins.lower() == 'yes',
        "tail": tail.lower() == 'yes',
        "legs": int(legs)
    }

    # Convert the dictionary to a DataFrame
    animal_data = pd.DataFrame([animal_info])

    # Make predictions using the trained model
    predictions = predict_model(tuned_et_model, data=animal_data)
    
    predicted_class = int(predictions["prediction_label"].values[0])
    
    # Define dictionary to map predicted class to animal name
    animal_class = {
        1: "Mammal",
        2: "Bird",
        3: "Reptile",
        4: "Fish",
        5: "Amphibian",
        6: "Bug",
        7: "Invertebrate"
    }

    # Loop through the dictionary to find the animal name corresponding to the predicted class
    for class_num, animal_class in animal_class.items():
        if predicted_class == class_num:
            predicted_animal_name = animal_class
            break

    # Return the predicted animal name to the client
    message = f' Congratulation, I was able to predict {animal_name}s Classification. {animal_name} is indeed a {predicted_animal_name}'.upper()

    return message
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)




#client = MongoClient("mongodb://localhost:27017/")
#db = client["animal_database"]
#collection = db["animals"]

#data = zoolive.to_dict(orient='records')
#collection.insert_many(data)


#result = collection.insert_one(animal_data)

#zoolive = zoolive.to_dict('records')

    

#client.close()



