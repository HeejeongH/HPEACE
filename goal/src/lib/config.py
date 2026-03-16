# LLM API
GOOGLE_API_KEY = ''
API_MODEL =  'gemini-2.5-flash' # gemini-2.5-pro
MAX_RETRY = 10

diet_cols = [
    'Meal Frequency', 'Meal Portion Size', 'Eating Out Frequency', 'Rice Portion Size', 
    'Snacking Frequency', 'Grain Products', 'Protein Foods', 'Vegetables', 'Dairy Products', 
    'Fruits', 'Fried Foods', 'High Fat Meat', 'Processed Foods', 'Water Intake', 
    'Coffee Consumption', 'Sugar-Sweetened Beverages', 'Additional Salt Use', 
    'Salty Food Consumption', 'Sweet Food Consumption'
]

life_cols = ['Physical activity', 'Alcohol Consumption', 'Current smoking']

conf_cols = ['Age', 'Sex', 'Education Level', 'Marital Status', 'Household Income']

disease_cols = [
    'Increased waist circumference', 'Elevated blood pressure',
    'Impaired fasting glucose', 'Elevated triglycerides', 'Decreased HDL-C',
    'MetS'
]
diseases_value_cols = [
    'Waist circumference (cm)',
    'Systolic blood pressure (mmHg)',
    'Fasting glucose (mg/dL)',
    'Triglycerides (mg/dL)',
    'HDL-C (mg/dL)'
]

target_goal_cols = ['Cluster', 'Physical activity', 'Alcohol Consumption', 'Current smoking']

checkpoint_path = '../results/clustering_result/reproduction/best_model.pth'
artifact_path = '../results/clustering_result/reproduction/MetS_LR_SHAP_Artifact.pkl'

json_data_path = '../data/json'
