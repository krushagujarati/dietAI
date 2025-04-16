from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load encoders (make sure you trained & saved these earlier)
le_gender = joblib.load('le_gender.pkl')
le_activity = joblib.load('le_activity.pkl')
le_goal = joblib.load('le_goal.pkl')
le_preference = joblib.load('le_preference.pkl')
le_lifestyle = joblib.load('le_lifestyle.pkl')
le_restriction = joblib.load('le_restriction.pkl')
le_health_condition = joblib.load('le_health_condition.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        age = int(request.form['age'])
        gender_raw = request.form.get('gender')
        height = int(request.form['height'])
        weight = int(request.form['weight'])
        activity_raw = request.form.get('activity')
        goal_raw = request.form.get('goal')
        preference_raw = request.form.get('preference')
        lifestyle_raw = request.form.get('lifestyle')
        restriction_raw = request.form.get('restriction')
        health_condition_raw = request.form.get('health_condition')

        # Check if all fields are filled
        if any(val in ['', None] for val in [gender_raw, activity_raw, goal_raw, preference_raw, lifestyle_raw, restriction_raw, health_condition_raw]):
            return render_template('error.html', message="Please fill in all fields.")

        # Encode categories, treating "None" as a valid option
        gender = le_gender.transform([gender_raw])[0]
        activity = le_activity.transform([activity_raw])[0]
        goal = le_goal.transform([goal_raw])[0]
        preference = le_preference.transform([preference_raw])[0] if preference_raw != 'None' else -1  # Assuming -1 for None
        lifestyle = le_lifestyle.transform([lifestyle_raw])[0]
        restriction = le_restriction.transform([restriction_raw])[0]
        health_condition = le_health_condition.transform([health_condition_raw])[0] if health_condition_raw != 'None' else -1  # Assuming -1 for None

        # Predict
        input_data = np.array([[age, gender, height, weight, activity, goal, preference, lifestyle, restriction, health_condition]])
        model = joblib.load('calorie_predictor_model.pkl')
        predicted_calories = model.predict(input_data)[0]

        return render_template('result.html', calories=int(predicted_calories))

    except Exception as e:
        return render_template('error.html', message=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5050)
