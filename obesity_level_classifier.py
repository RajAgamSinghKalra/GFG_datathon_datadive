import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# load kara dataset ko file se, put file same folder m
data_frame = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

for column in data_frame.select_dtypes(include=['object']).columns:
    data_frame[column] = data_frame[column].str.capitalize()

encoders_for_columns = {}
for column in data_frame.select_dtypes(include=['object']).columns:
    label_encoder = LabelEncoder()
    data_frame[column] = label_encoder.fit_transform(data_frame[column])
    encoders_for_columns[column] = label_encoder

features = data_frame.drop('NObeyesdad', axis=1)
target = data_frame['NObeyesdad']

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
forest_model.fit(features_train, target_train)

# bhavishya wani
predictions = forest_model.predict(features_test)
print(classification_report(target_test, predictions))

# Drawing insights, literally
def draw_diet_chart(observation):
    # Pizza chart 
    diet_labels = ['High Caloric Food', 'Vegetables', 'Healthy Meals', 'Snacks']
    diet_portions = [1 if observation[5] == 'Yes' else 0, 
                     observation[6], 
                     observation[7], 
                     1 if observation[8] != 'No' else 0]
    diet_colors = ['gold', 'lightgreen', 'lightcoral', 'skyblue']
    explode_first_slice = (0.1, 0, 0, 0)  # pasta pizza slice
    plt.pie(diet_portions, explode=explode_first_slice, labels=diet_labels, colors=diet_colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # gol gol
    plt.title("Your Diet Habits")
    plt.show()

    # bar waala chart
    activity_labels = ['Physical Activity', 'Tech Use']
    activity_hours = [float(observation[12]), int(observation[13])]
    plt.bar(activity_labels, activity_hours, color=['forestgreen', 'tomato'])
    plt.ylabel('Hours per Week')
    plt.title('Your Activity Overview')
    plt.show()

# ptla hone ki advice
def mota_advice(observation):
    advice_list = []
    if observation[5] == "Yes":
        advice_list.append("Cutting down on high-calorie food might be a good idea.")
    if int(observation[6]) < 3:
        advice_list.append("How about more veggies in your diet?")
    if int(observation[7]) != 3:
        advice_list.append("Eating regular meals could help with weight management.")
    if observation[8] != "No":
        advice_list.append("Maybe snack a little less?")
    if observation[9] == "Yes":
        advice_list.append("Quitting smoking is tough but very beneficial.")
    if int(observation[10]) < 3:
        advice_list.append("Drinking more water is always a good idea.")
    if float(observation[12]) < 2:
        advice_list.append("A bit more physical activity could do wonders.")
    if int(observation[13]) >= 2:
        advice_list.append("Perhaps cut down on screen time?")
    if observation[14] != "No":
        advice_list.append("Less alcohol could be beneficial for your health.")
    
    return advice_list if advice_list else ["You're on the right track! Keep it up."]

# model ka input kardo
def model_ke_liye_input(observation, column_encoders, columns_in_data):
    processed_observation = {}
    for column_name, value in zip(columns_in_data, observation):
        if column_name in column_encoders:
            value = value.capitalize()
            encoder = column_encoders[column_name]
            if value not in encoder.classes_:
                encoder.classes_ = np.append(encoder.classes_, value)
            processed_observation[column_name] = encoder.transform([value])[0]
        else:
            processed_observation[column_name] = value
    return pd.DataFrame([processed_observation], columns=columns_in_data)

# input
def user_ka_input(column_names):
    print("Let's talk about you. Don't worry, it's all confidential:")
    user_info = []
    for column in column_names:
        if 'yes' in column.lower() or 'no' in column.lower():
            prompt = f"{column} : "
        else:
            prompt = f"{column} : "
        user_info.append(input(prompt).capitalize())
    return user_info

# output
column_names_list = features.columns.tolist()
user_data = user_ka_input(column_names_list)
prepared_data = model_ke_liye_input(user_data, encoders_for_columns, column_names_list)

obesity_level_prediction = forest_model.predict(prepared_data)[0]
obesity_level = encoders_for_columns['NObeyesdad'].inverse_transform([obesity_level_prediction])[0]
print(f"Your predicted obesity level is: {obesity_level}")

draw_diet_chart(user_data)

health_advice = mota_advice(user_data)
if health_advice:
    print("Based on what you've told me, here are some friendly suggestions:")
    for single_advice in health_advice:
        print("- " + single_advice)
else:
    print("Looks like you're doing great! Keep it up.")