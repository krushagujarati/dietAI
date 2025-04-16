from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load encoders
le_gender = joblib.load('le_gender.pkl')
le_activity = joblib.load('le_activity.pkl')
le_goal = joblib.load('le_goal.pkl')
le_preference = joblib.load('le_preference.pkl')
le_lifestyle = joblib.load('le_lifestyle.pkl')
le_restriction = joblib.load('le_restriction.pkl')
le_health_condition = joblib.load('le_health_condition.pkl')

# Sample recipe and workout data
recipes_data = {
    "vegan": [
        {"title": "Chickpea Quinoa Bowl", "calories": 450},
        {"title": "Tofu Stir-Fry", "calories": 400}
    ],
    "vegetarian": [
        {"title": "Paneer Tikka Bowl", "calories": 500},
        {"title": "Methi Thepla with Curd", "calories": 350}
    ],
    "non_vegetarian": [
        {"title": "Grilled Chicken with Rice", "calories": 550},
        {"title": "Egg Curry with Brown Rice", "calories": 500}
    ]
}

workouts_data = {
    "weight_loss": [
        "30 min Yoga + Light walk",
        "Strength training (Upper body)",
        "HIIT + Core workout"
    ],
    "weight_gain": [
        "Heavy weight training (Upper body)",
        "Lower body muscle building",
        "Bodyweight + Protein focus"
    ],
    "maintain_weight": [
        "Yoga + Strength",
        "Jogging + Bodyweight",
        "Pilates or Swimming"
    ]
}

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

        # Encode categories
        gender = le_gender.transform([gender_raw])[0]
        activity = le_activity.transform([activity_raw])[0]
        goal = le_goal.transform([goal_raw])[0]
        preference = le_preference.transform([preference_raw])[0] if preference_raw != 'None' else -1
        lifestyle = le_lifestyle.transform([lifestyle_raw])[0]
        restriction = le_restriction.transform([restriction_raw])[0]
        health_condition = le_health_condition.transform([health_condition_raw])[0] if health_condition_raw != 'None' else -1

        # Predict calories
        input_data = np.array([[age, gender, height, weight, activity, goal, preference, lifestyle, restriction, health_condition]])
        model = joblib.load('calorie_predictor_model.pkl')
        predicted_calories = model.predict(input_data)[0]
        calories = int(predicted_calories)

        # Macronutrient breakdown
        macros = {
            "protein": int((calories * 0.27) / 4),
            "carbs": int((calories * 0.45) / 4),
            "fats": int((calories * 0.28) / 9)
        }

        # Get recipe suggestions
        preference_key = le_preference.inverse_transform([preference])[0].lower().replace(" ", "_") if preference != -1 else 'vegetarian'
        recipes = recipes_data.get(preference_key, [])

        # Get workouts
        goal_key = le_goal.inverse_transform([goal])[0].lower().replace(" ", "_")
        workouts = workouts_data.get(goal_key, [])

        return render_template('result.html',
                               calories=calories,
                               macros=macros,
                               recipes=recipes,
                               workouts=workouts)

    except Exception as e:
        return render_template('error.html', message=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5050)
