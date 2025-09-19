import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle

# Load dataset from local file (update path as needed)
df = pd.read_csv(r'D:\Projects\fast-api ml\insurance.csv')

# Add computed feature: BMI
df['bmi'] = df['weight'] / (df['height'] ** 2)

# Define functions to classify age group, lifestyle risk, and city tier as per your app's logic

tier_1_cities = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"
]

tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
]

def age_group(age):
    if age < 25:
        return "young"
    elif age < 45:
        return "adult"
    elif age < 60:
        return "middle_aged"
    else:
        return "senior"

def lifestyle_risk(row):
    bmi = row['bmi']
    smoker = row['smoker']
    if smoker and bmi > 30:
        return "high"
    elif smoker or bmi > 27:
        return "medium"
    else:
        return "low"

def city_tier(city):
    if city in tier_1_cities:
        return 1
    elif city in tier_2_cities:
        return 2
    else:
        return 3

# Apply computed features
df['age_group'] = df['age'].apply(age_group)
df['lifestyle_risk'] = df.apply(lifestyle_risk, axis=1)
df['city_tier'] = df['city'].apply(city_tier)

# Define features and target
features = ['age_group', 'lifestyle_risk', 'occupation', 'city_tier', 'bmi', 'income_lpa']
target = 'insurance_premium_category'

X = df[features]
y = df[target]

# Preprocessing for categorical and numerical features
categorical_features = ['age_group', 'lifestyle_risk', 'occupation', 'city_tier']
numeric_features = ['bmi', 'income_lpa']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

# Create pipeline with preprocessing and classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
pipeline.fit(X, y)

# Save the trained model to 'model.pkl'
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model training complete and saved to model.pkl")
