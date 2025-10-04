import streamlit as st
from streamlit_lottie import st_lottie
import json
import pandas as pd
import numpy as np
from PIL import Image
import joblib
import plotly.graph_objects as go


# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="BayMax - Personal Fitness Guide", page_icon="💪")

# -------------------------
# Load Lottie animation
# -------------------------
with open("robot.json", "r", encoding="utf-8") as f:
    baymax_animation = json.load(f)
st_lottie(baymax_animation, height=300, key="baymax")

st.title("👋 Hi, I'm BayMax, your personal fitness guide!")
st.write("Let's get started with your health profile. Please fill in the details below 👇")

# -------------------------
# User Inputs
# -------------------------
name = st.text_input("What's your name?")
age = st.number_input("Age", min_value=12, max_value=80, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, step=0.1)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, step=0.1)

exercise = st.slider("Exercise per week (hours)", 0, 20, 3)
smoking = st.radio("Do you smoke?", ["Yes", "No"])
drinking = st.radio("Do you drink alcohol?", ["Yes", "No"])
screen_time = st.slider("Average screen time per day (hours)", 0, 16, 5)
sleep = st.slider("Average sleep per day (hours)", 0, 12, 7)

goal = st.selectbox("What's your primary fitness goal?", 
                    ["Lose weight", "Gain weight", "Maintain weight", "Build strength", "Improve stamina"])

# -------------------------
# Load trained model
# -------------------------
model = joblib.load("health_risk_model.pkl")

# -------------------------
# Helper Functions
# -------------------------
def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    return weight / (height_m ** 2)

def bmi_classification(bmi):
    if bmi < 18.5:
        return "underweight"
    elif 18.5 <= bmi <= 24.9:
        return "normal_weight"
    elif 25 <= bmi <= 29.9:
        return "overweight"
    elif 30 <= bmi <= 39.9:
        return "obesity"
    else:
        return "extreme_obesity"

def calculate_bmr(weight, height, age, gender_val):
    if gender_val == 0:  # Male
        return 10*weight + 6.25*height - 5*age + 5
    else:  # Female
        return 10*weight + 6.25*height - 5*age - 161

def adjust_bmr_for_activity(bmr, exercise_hours):
    if exercise_hours < 3:
        factor = 1.2
    elif exercise_hours < 6:
        factor = 1.375
    elif exercise_hours < 10:
        factor = 1.55
    else:
        factor = 1.725
    return bmr * factor

def calculate_body_fat(bmi, age, gender_val):
    return round(1.2*bmi + 0.23*age - 10.8*gender_val - 5.4,2)

def plot_health_risk_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "Health Risk Score"},
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "white"},
                 'steps' : [
                     {'range': [0, 40], 'color': "green"},
                     {'range': [40, 70], 'color': "yellow"},
                     {'range': [70, 100], 'color': "red"}]}))
    st.plotly_chart(fig, use_container_width=True)

def plot_bmi_gauge(bmi):
    # Define gauge colors
    if bmi < 18.5:
        color = "lightblue"
    elif 18.5 <= bmi <= 24.9:
        color = "green"
    elif 25 <= bmi <= 29.9:
        color = "orange"
    elif 30 <= bmi <= 39.9:
        color = "red"
    else:
        color = "darkred"
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = bmi,
        title = {'text': "BMI"},
        gauge = {
            'axis': {'range':[10,50]},
            'steps': [
                {'range':[10,18.5],'color':'lightblue'},
                {'range':[18.5,24.9],'color':'green'},
                {'range':[25,29.9],'color':'orange'},
                {'range':[30,39.9],'color':'red'},
                {'range':[40,50],'color':'darkred'}
            ],
            'bar': {'color': 'white'}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def show_bmi_illustration(gender_str, bmi_cls, width=200):
    """
    Display BMI illustration centered with optional width
    """
    file_name = f"assets/{bmi_cls}_{gender_str.lower()}.png"
    img = Image.open(file_name)
    
    # Resize while maintaining aspect ratio
    w_percent = (width / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(w_percent)))
    
    # Use the correct resampling method for Pillow >=10
    try:
        resample_method = Image.Resampling.LANCZOS
    except AttributeError:
        resample_method = Image.ANTIALIAS  # fallback for older versions
    
    img = img.resize((width, h_size), resample_method)
    
    # Display centered using Streamlit columns
    col1, col2, col3 = st.columns([1,2,1])  # middle column wider
    with col2:
        st.image(img, caption=f"{bmi_cls.replace('_',' ').title()}", use_container_width=False)


def plot_macros(weight):
    protein = round(weight * 1.5)
    carbs = round(weight * 2.5)
    fat = round(weight * 0.8)
    fiber = 30 if gender.lower() == "male" else 25

    fig = go.Figure(data=[go.Bar(
        x=['Protein', 'Carbs', 'Fat', 'Fiber'],
        y=[protein, carbs, fat, fiber],
        marker_color=['blue', 'orange', 'green', 'purple']
    )])

    fig.update_layout(
        title="Daily Macro Requirements (grams)",
        yaxis_title="Grams",
        xaxis_title="Nutrients"
    )

    st.plotly_chart(fig, use_container_width=True)

def generate_exercise_table(bmi_cls, goal):
    # Define exercise pools by BMI class
    exercise_pool = {
        "underweight": [["Push-ups", "Squats", "Plank"],
                        ["Lunges", "Burpees", "Mountain Climbers"],
                        ["Pull-ups (assisted)", "Dips", "Crunches"]],
        
        "normal_weight": [["Push-ups", "Squats", "Plank"],
                          ["Lunges", "Burpees", "Jumping Jacks"],
                          ["Pull-ups", "Dips", "Crunches"]],
        
        "overweight": [["Brisk Walk (10 min)", "Bodyweight Squats", "Wall Push-ups"],
                       ["Step-ups", "Plank (hold)", "Glute Bridge"],
                       ["Modified Burpees", "Chair Squats", "Side Leg Raises"]],
        
        "obesity": [["Brisk Walk (10–15 min)", "Chair Squats", "Wall Push-ups"],
                    ["Step-ups (low)", "Marching in Place", "Seated Knee Lifts"],
                    ["Arm Raises", "Seated Leg Lifts", "Side Steps"]],
        
        "extreme_obesity": [["Chair Squats", "Wall Push-ups", "Arm Raises"],
                            ["Seated Leg Lifts", "Seated March", "Resistance Band Rows"],
                            ["Step-ups (very low)", "Marching in Place", "Overhead Stretch"]]
    }

    # Default sets/reps by goal
    goal_reps = {
        "Lose weight": "3x12–15",
        "Gain weight": "4x10–12",
        "Maintain weight": "3x12",
        "Build strength": "4x8–10",
        "Improve stamina": "3x15–20"
    }

    # Construct weekly plan (Mon–Sun, with rest days)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    plan = []
    pool = exercise_pool[bmi_cls]

    for i, day in enumerate(days):
        if day in ["Wed", "Sun"]:  # Rest days
            plan.append([day, "Rest Day", "-"])
        else:
            exercises = pool[i % len(pool)]
            for ex in exercises:
                plan.append([day, ex, goal_reps[goal]])

    df = pd.DataFrame(plan, columns=["Day", "Exercise", "Sets x Reps"])
    return df

def display_weekly_exercise_plan(bmi_cls, goal):
    # Generate plan as before (3 exercises per workout day + rest days)
    df = generate_exercise_table(bmi_cls, goal)

    # Group by day
    days = df['Day'].unique()
    for day in days:
        day_df = df[df['Day'] == day].copy()
        with st.expander(day):
            # For rest day, show message
            if day_df.iloc[0]['Exercise'] == "Rest Day":
                st.write("🛌 Rest Day")
            else:
                # Show exercises for the day
                st.table(day_df[['Exercise', 'Sets x Reps']].reset_index(drop=True))

import requests
import json

def get_ai_suggestion(prompt_text):
    url = "https://apifreellm.com/api/chat"
    headers = {"Content-Type": "application/json"}
    data = {"message": prompt_text}

    try:
        response = requests.post(url, headers=headers, json=data, timeout=20)
        result = response.json()

        if result.get("status") == "success":
            return result["response"]
        else:
            return f"Error: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Request failed: {e}"

if st.button("Submit"):
    st.success(f"Thanks {name}! 🎉 Your profile has been saved.")
    
    # Convert gender to int for calculations
    gender_val = 0 if gender=="Male" else 1
    
    # -------------------------
    # Prepare DataFrame for model
    # -------------------------
    user_df = pd.DataFrame([{
        "age": age,
        "weight": weight,
        "height": height,
        "gender": gender_val,
        "exercise_hours": exercise,
        "smoking": 1 if smoking=="Yes" else 0,
        "drinking": 1 if drinking=="Yes" else 0,
        "screen_time": screen_time,
        "sleep_hours": sleep
    }])
    
    # -------------------------
    # Predict health risk score
    # -------------------------
    risk_score = model.predict(user_df)[0]
    st.subheader("🩺 Health Risk Score")
    plot_health_risk_gauge(risk_score)
    
    # -------------------------
    # BMI
    # -------------------------
    bmi = calculate_bmi(weight, height)
    bmi_cls = bmi_classification(bmi)
    st.subheader("⚖️ BMI")
    st.write(f"BMI: {bmi:.2f} ({bmi_cls.replace('_',' ').title()})")

    # Plot interactive BMI gauge
    plot_bmi_gauge(bmi)

    # Centered BMI illustration only
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        show_bmi_illustration(gender, bmi_cls, width=200)
    
    # -------------------------
    # BMR & Calories
    # -------------------------
    bmr = calculate_bmr(weight, height, age, gender_val)
    adj_calories = adjust_bmr_for_activity(bmr, exercise)
    st.subheader("🔥 BMR & Daily Calories")
    st.write(f"BMR: {bmr:.0f} kcal")
    st.write(f"Adjusted for activity: {adj_calories:.0f} kcal/day")
    
    # -------------------------
    # Body Fat %
    # -------------------------
    body_fat = calculate_body_fat(bmi, age, gender_val)
    st.subheader("💧 Body Fat Percentage")
    st.write(f"Estimated Body Fat: {body_fat}%")
    
    # -------------------------
    # Nutrition
    # -------------------------
    st.subheader("🥗 Nutrition & Macros")
    plot_macros(weight)
    
    # -------------------------
    # Exercise Table
    # -------------------------
    st.subheader("🏋️ Weekly Exercise Plan")
    display_weekly_exercise_plan(bmi_cls, goal)

    # -------------------------
    # ChatGPT Suggestions
    # -------------------------
st.subheader("💡 AI Lifestyle Suggestions")

prompt = (
    f"Hey {name}! Based on the details you provided — {age} years old, {gender}, "
    f"{height} cm tall, weighing {weight} kg, exercising {exercise} hours per week, "
    f"{'smoking' if smoking == 'Yes' else 'not smoking'}, "
    f"{'drinking alcohol' if drinking == 'Yes' else 'not drinking'}, "
    f"{screen_time} hours of screen time per day, and sleeping {sleep} hours per day, "
    f"with your main fitness goal being '{goal}' — give 4-5 personalized, actionable, "
    f"and motivating health and fitness tips directly for {name}. "
    f"Keep the tone friendly, easy to follow, and concise, as if you're talking to {name} personally."
)

suggestion = get_ai_suggestion(prompt)
st.info(suggestion)

