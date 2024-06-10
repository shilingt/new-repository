import streamlit as st
import pandas as pd
import math
from pathlib import Path
import datetime

import numpy as np
import pickle
from joblib import load
from sklearn.preprocessing import LabelEncoder, StandardScaler
      
    
# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Exercise Prescription Using Cardiac Risk Level',
    page_icon= 'https://arcadiapittwater.com.au/wp-content/uploads/2022/07/Cardiopulmonary-day-rehabilitation.png', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

# Define the Streamlit app


# Load the trained model
risk_scaler = load('risk_scaler.pkl')

# Function to preprocess user input data
def preprocess_data(data):
    # Create a dataframe from the input data
    
    today = datetime.date.today()
    
    ## Add columns
    df = pd.DataFrame(data)
    df['Year'] = today.year
    df['Total Muscle Power - UL'] = df['Muscle Power - UL - Right'] + df['Muscle Power - UL - Left']
    df['Total Muscle Power - LL'] = df['Muscle Power - LL - Right'] + df['Muscle Power - LL - Left']
    df['Total Muscle Power - Right'] = df['Muscle Power - UL - Right'] + df['Muscle Power - LL - Right']
    df['Total Muscle Power - Left'] = df['Muscle Power - UL - Left'] + df['Muscle Power - LL - Left']
    df['Total Muscle Power'] = df['Total Muscle Power - Right'] + df['Total Muscle Power - Left']
    
    df['Total Exercise Duration'] = df['Exercise Habit - Frequency'] * df['Exercise Habit - Duration']

    ## One Hot Encoding 1
    numeric_feature_names=['Year', 'Age', 'Exercise Habit - Frequency', 'Exercise Habit - Duration', 'Muscle Power - LL - Left',
                       'Muscle Power - LL - Right','Muscle Power - UL - Right','Muscle Power - UL - Left',
                       'Total Exercise Duration', 'Total Muscle Power - UL', 'Total Muscle Power - LL', 
                       'Total Muscle Power - Right', 'Total Muscle Power - Left','Total Muscle Power']

    special_categorical_columns = ['Exercise Habit - Mode','Test Today - Termination Cause','ECG Resting', 'Diagnosis']

    def one_hot_encode_categories(df, column_name, categories):
        encoded_df = pd.DataFrame()

        # Split by '+' and expand into separate columns
        split_columns = df2[column_name].str.split('+', expand=True)

        # Iterate over each category and encode it
        for category in categories:
            # Create a new column for the category and set values based on presence
            encoded_df[f'{column_name}_{category}'] = df2[column_name].apply(lambda x: 1 if category in str(x).split('+') else 0)

        return encoded_df
    
    df2= df.drop(columns=numeric_feature_names)
    df2= df2.drop(columns=special_categorical_columns)

    normal_columns = df2.columns.tolist()

    columns_to_keep = numeric_feature_names + special_categorical_columns
    df = df[columns_to_keep]

    # Categories to one-hot encode
    Gender_type = ['F', 'M']
    Marital_type =  ['divorced', 'married', 'single', 'widow']
    Lives_type =  ['alone', 'family', 'friends']
    Occupation_type =  ['employed','not working', 'retired', 'self employed']
    Smoking_type =  ['ex smoker', 'no', 'yes']
    Family_type =  ['no', 'yes']
    ROM_type =  ['abnormal', 'normal']
    Balance_type =  ['no', 'yes']
    Functional_type =  ['assisted', 'independent ']
    Walking_type =  ['dependent', 'independent']
    Gait_type =  ['abnormal', 'normal']
    Posture_type =  ['abnormal', 'normal']
    HPT_type =  ['no', 'yes']
    DM_type =  ['no', 'yes']
    HPL_type =  ['no', 'yes']
    Exercise_type =  ['active', 'inactive', 'moderate']
    Stress_type =  ['no', 'yes']
    BMI_type =  ['healthy', 'obese', 'overweight','underweight']
    ECHO_type =  ['borderline','normal','reduced']
    TestHR_type =  [ 'above maximum intensity', 'high intensity', 'low intensity', 'maximum intensity', 'moderate intensity']
    TestMETS_type =  ['high intensity', 'low intensity',  'moderate intensity (high)', 'moderate intensity (low)' ]

    # Perform one-hot encoding
    gender_encoded = one_hot_encode_categories(df2, 'Gender', Gender_type)
    marital_encoded = one_hot_encode_categories(df2, 'Marital Status', Marital_type)
    lives_encoded = one_hot_encode_categories(df2, 'Lives With', Lives_type)
    occupation_encoded = one_hot_encode_categories(df2, 'Occupation', Occupation_type)
    smoking_encoded = one_hot_encode_categories(df2, 'Smoking', Smoking_type)
    family_encoded = one_hot_encode_categories(df2, 'Family History', Family_type)
    rom_encoded = one_hot_encode_categories(df2, 'ROM', ROM_type)
    balance_encoded = one_hot_encode_categories(df2, 'Balance in Sitting and Standing', Balance_type)
    functional_encoded = one_hot_encode_categories(df2, 'Functional Activity', Functional_type)
    walking_encoded = one_hot_encode_categories(df2, 'Walking', Walking_type)
    gait_encoded = one_hot_encode_categories(df2, 'Gait', Gait_type)
    posture_encoded = one_hot_encode_categories(df2, 'Posture', Posture_type)
    hpt_encoded = one_hot_encode_categories(df2, 'Risk Factor - HPT', HPT_type)
    dm_encoded = one_hot_encode_categories(df2, 'Risk Factor - DM', DM_type)
    hpl_encoded = one_hot_encode_categories(df2, 'Risk Factor - HPL', HPL_type)
    exercise_encoded = one_hot_encode_categories(df2, 'Risk Factor - Exercise', Exercise_type)
    stress_encoded = one_hot_encode_categories(df2, 'Risk Factor - Stress', Stress_type)
    bmi_encoded = one_hot_encode_categories(df2, 'Risk Factor - BMI', BMI_type)
    echo_encoded = one_hot_encode_categories(df2, 'Risk Factor - ECHO - EF', ECHO_type)
    testhr_encoded = one_hot_encode_categories(df2, 'Test Today - peak HR', TestHR_type)
    testmets_encoded = one_hot_encode_categories(df2, 'Test Today - METS', TestMETS_type)

    # Concatenate the one-hot encoded DataFrames along the columns axis
    one_hot_encoded_df = pd.concat([df2, gender_encoded, marital_encoded, lives_encoded, occupation_encoded, smoking_encoded, family_encoded,
                             rom_encoded, balance_encoded, functional_encoded, walking_encoded, gait_encoded, posture_encoded,
                             hpt_encoded, dm_encoded, hpl_encoded, exercise_encoded, stress_encoded, bmi_encoded,
                             echo_encoded, testhr_encoded, testmets_encoded], axis=1)

    # Concatenate the one-hot encoded DataFrames along the columns axis
    merged_df = pd.concat([df, one_hot_encoded_df], axis=1)
    merged_df = merged_df.drop(columns=normal_columns)
    
    ## One Hot Encoding 2
    # One Hot Encoding special categorical columsn

    df3 = df[special_categorical_columns]
    df  = df.drop(columns= special_categorical_columns)

    def one_hot_encode_categories(df, column_name, categories):
        encoded_df = pd.DataFrame()

        # Split by '+' and expand into separate columns
        split_columns = df3[column_name].str.split('+', expand=True)

        # Iterate over each category and encode it
        for category in categories:
            # Create a new column for the category and set values based on presence
            encoded_df[f'{column_name}_{category}'] = df3[column_name].apply(lambda x: 1 if category in str(x).split('+') else 0)

        return encoded_df

    # Categories to one-hot encode
    modes = ['walking', 'jogging', 'cycling', 'no', 'others']
    termination_causes = ['Complete Test', 'Fatigue', 'Medical Condition', 'Physical Discomfort']
    ecg_resting_states = ['normal', 'sinus rhythm', 'T wave inversion', 'st depression', 'Q wave', 'Ectopics']
    diagnosis_status = ['PCI', 'CABG', 'conservative', 'surgical']

    # Perform one-hot encoding
    modes_encoded = one_hot_encode_categories(df3, 'Exercise Habit - Mode', modes)
    termination_cause_encoded = one_hot_encode_categories(df3, 'Test Today - Termination Cause', termination_causes)
    ecg_resting_encoded = one_hot_encode_categories(df3, 'ECG Resting', ecg_resting_states)
    diagnosis_encoded = one_hot_encode_categories(df3, 'Diagnosis', diagnosis_status)


    # Concatenate the one-hot encoded DataFrames along the columns axis
    merged_df = pd.concat([merged_df, modes_encoded, termination_cause_encoded, ecg_resting_encoded,diagnosis_encoded], axis=1)
    merged_df = merged_df.drop(columns=special_categorical_columns)

    df = merged_df
        
    
    # Replace NaN values with mean
    df.fillna(0, inplace=True)

    ## Remove space and replace '_' to '-'

    # Iterate over all columns and rename them by removing spaces and symbols '<', '>', '='
    for old_col in df.columns:
        new_col = old_col.replace(' ', '').replace('<', '').replace('>', '').replace('=', '')
        df.rename(columns={old_col: new_col}, inplace=True)

    # Identify string columns
    string_columns = df.select_dtypes(include=['object']).columns

    # Replace spaces with hyphens in string columns only
    df[string_columns] = df[string_columns].apply(lambda x: x.str.replace(' ', '-'))

    # Replace spaces with hyphens in string columns only
    df[string_columns] = df[string_columns].apply(lambda x: x.str.replace('+', '-'))
 
    
    ########## Processed Risk Level Data #######################
    
    target_variable = 'RiskLevel' #change 4

    key_feature_Risk_Level = ['TotalExerciseDuration', 'ExerciseHabit-Frequency', 'Age', 'FamilyHistory', 'TestToday-METS', 'Year', 
                              'TestToday-TerminationCause', 'RiskFactor-Exercise', 'ExerciseHabit-Duration', 'ECGResting', 'Occupation', 
                              'TestToday-peakHR', 'Diagnosis', 'RiskFactor-BMI', 'RiskFactor-HPT', 'RiskFactor-ECHO-EF', 'ExerciseHabit-Mode', 
                              'Smoking', 'RiskFactor-DM', 'Gender', 'RiskFactor-HPL', 'RiskFactor-Stress', 'TotalMusclePower', 'Walking', 
                              'MaritalStatus', 'FunctionalActivity', 'TotalMusclePower-Right', 'ROM', 'BalanceinSittingandStanding', 'Posture', 
                              'MusclePower-UL-Left', 'Gait', 'LivesWith', 'MusclePower-LL-Left']

    key_features = key_feature_Risk_Level #change 6    
    
    
    # Initialize a dictionary to hold the lists of relevant columns for each key feature
    relevant_columns = {}

    # Initialize a list to hold all relevant column names
    key_features_columns = []

    # Iterate through each key feature
    for feature in key_feature_Risk_Level:                                                        #change 8
        # Filter the columns that match the key feature exactly before any underscore
        # and include columns that extend beyond the key feature name with an underscore
        # also include columns that match the original feature name without underscores
        filtered_columns = [column for column in df if column.split('_')[0] == feature or 
                            column.startswith(f"{feature}_") or 
                            column == feature]  # New condition added here
        # Extend the key_features_columns list with the filtered columns
        key_features_columns.extend(filtered_columns)  

        # Limit to key feature only
        df2 = df[key_features_columns]
        
        X_test = df2
        X_test_scaled = risk_scaler.transform(X_test)

        risk_X_test_scaled = X_test_scaled
    
    return risk_X_test_scaled




def risk_ensemble_predict(X_test_scaled):
    import pickle
    from scipy.stats import mode
    import numpy as np
    
    from joblib import load
    from scipy.stats import mode
    import numpy as np

    rf_model = load('risk_randomforest.pkl')
    bbc_model = load('risk_bbc.pkl')
    lr_model = load('risk_logistic.pkl')
    label_encoder = load('risk_label_encoder.pkl')

    # Make predictions
    rf_preds = rf_model.predict(X_test_scaled)
    bbc_preds = bbc_model.predict(X_test_scaled)
    lr_preds = lr_model.predict(X_test_scaled)

    # Majority voting
    preds = np.array([rf_preds, bbc_preds, lr_preds])
    majority_vote_preds = mode(preds, axis=0).mode
    
    # Decode the integer predictions back to original labels
    risk_level_startification = label_encoder.inverse_transform(majority_vote_preds)


    return risk_level_startification

# Load the trained model
risk_ensemble_model = load('risk_ensemble.pkl')




# Define the Streamlit app
def main():
    st.title("Cardiac Risk Predictor")

    st.markdown(" ##### Demographic")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        Gender = st.selectbox('Gender', ['M', 'F'])
        Age = st.number_input('Age', min_value=1, max_value=120, value=45)
    with col2:
        MaritalStatus = st.selectbox('Marital Status', ['married', 'single', 'divorced', 'widow'])
        LivesWith = st.selectbox('Lives With', ['family', 'friends', 'alone '])
    with col3:
        Occupation = st.selectbox('Occupation', ['employed', 'self employed','not working', 'retired'])
        FamilyHistory = st.selectbox('Family History', ['yes', 'no'])
    

    st.markdown(" ##### Lift Style")
    col1, col2, col3, col4 = st.columns(4)
    with col1:    
        Stress = st.selectbox('Stress', ['yes', 'no'])
        ExerciseHabitMode = st.multiselect('Exercise Mode', ['walking', 'cycling', 'jogging', 'others', 'no'], default=['walking'])
    with col2:
        Smoking = st.selectbox('Smoking', ['ex smoker ', 'no', 'yes'])
        ExerciseHabitDuration = st.number_input('Exercise Duration (# of minutes per frequency)', min_value=0, max_value=480, value=30)
    with col3:
        Exercise = st.selectbox('Exercise', ['active', 'moderate', 'inactive'])
        ExerciseHabitFrequency = st.number_input('Exercise Frequency (# of times weekly)', min_value=0, max_value=14, value=2)
    
    
    st.markdown(" ##### Muscle Strength and Balancing")
    col1, col2, col3 = st.columns(3)
    with col1:
        MusclePowerULRight = st.number_input('Muscle Power - Upper Limb - Right', min_value=0, max_value=5, value=5)
        MusclePowerULLeft = st.number_input('Muscle Power - Upper Limb - Left', min_value=0, max_value=5, value=5)
        MusclePowerLLRight = st.number_input('Muscle Power - Lower Limb - Right', min_value=0, max_value=5, value=5)
        MusclePowerLLLeft = st.number_input('Muscle Power - Lower Limb - Left', min_value=0, max_value=5, value=5)
    with col2:
        ROM = st.selectbox('Range Of Motion', ['abnormal', 'normal'])
        BalanceinSittingandStanding = st.selectbox('Balance in Sitting and Standing', ['yes', 'no'])
        FunctionalActivity = st.selectbox('Functional Activity', ['independent ', 'assisted'])
    with col3:
        Walking = st.selectbox('Walking', ['independent', 'dependent'])
        Gait = st.selectbox('Gait', ['normal', 'abnormal'])
        Posture = st.selectbox('Posture', ['normal', 'abnormal'])

    st.markdown(" ##### Medical Conditions")
    col1, col2, col3 = st.columns(3)
    with col1:    
        HPT = st.selectbox('HPT', ['yes', 'no'])
        DM = st.selectbox('DM', ['yes', 'no'])
        HPL = st.selectbox('HPL', ['yes', 'no'])
    with col2:
        BMI = st.selectbox('BMI', ['underweight', 'healthy', 'overweight', 'obese'])
        ECHOEF = st.selectbox('ECHO - EF', ['normal', 'borderline','reduced'])
    with col3:
        Diagnosis = st.multiselect('Diagnosis', ['PCI', 'conservative', 'CABG', 'surgical'], default=['PCI'])
        ECGResting = st.multiselect('ECG Resting', ['normal', 'sinus rhythm', 'T wave inversion',  'st depression', 'Q wave ectopics'], default=['normal'])
    
    st.markdown(" ##### Medical Conditions")
    col1, col2, col3 = st.columns(3)
    with col1:    
        TestTodayTerminationCause = st.multiselect('Test Today - Termination Cause', ['Complete Test', 'Fatigue', 'Medical Condition', 'Physical Discomfort'], default=['Complete Test'])
    with col2:
        TestTodaypeakHR = st.selectbox('Test Today - peak HR', ['low intensity', 'moderate intensity', 'high intensity', 'maximum intensity', 'above maximum intensity'])
    with col3:
        TestTodayMETS = st.selectbox('Test Today - METS', ['low intensity',  'moderate intensity (low)', 'moderate intensity (high)', 'high intensity' ])
        
    # Button for making prediction
    if st.button('Predict'):
        # Collect input data into a dictionary
        input_data = {
            'Gender': Gender,
            'Age': Age,
            'Marital Status': MaritalStatus,
            'Lives With': LivesWith,
            'Occupation': Occupation,
            'Smoking': Smoking,
            'Family History': FamilyHistory,
            'Exercise Habit - Frequency': ExerciseHabitFrequency,
            'Exercise Habit - Mode': ExerciseHabitMode,
            'Exercise Habit - Duration': ExerciseHabitDuration,
            'ROM': ROM,
            'Muscle Power - UL - Right': MusclePowerULRight,
            'Muscle Power - UL - Left': MusclePowerULLeft,
            'Muscle Power - LL - Right': MusclePowerLLRight,
            'Muscle Power - LL - Left': MusclePowerLLLeft,
            'Balance in Sitting and Standing': BalanceinSittingandStanding,
            'Functional Activity': FunctionalActivity,
            'Walking': Walking,
            'Gait': Gait,
            'Posture': Posture,
            'Risk Factor - HPT': HPT,
            'Risk Factor - DM': DM,
            'Risk Factor - HPL': HPL,
            'Risk Factor - Exercise': Exercise,
            'Risk Factor - Stress': Stress,
            'Risk Factor - BMI': BMI,
            'Risk Factor - ECHO - EF': ECHOEF,
            'Test Today - Termination Cause': '+'.join(TestTodayTerminationCause),
            'Test Today - peak HR': TestTodaypeakHR,
            'Test Today - METS': TestTodayMETS,
            'ECG Resting': '+'.join(ECGResting),
            'Diagnosis': '+'.join(Diagnosis)
        
        }

        # Preprocess the input data
        input_df = preprocess_data(input_data)

        # Make prediction
        prediction = risk_ensemble_predict(input_df)

        # Display the result below the button
        st.markdown("### Result")
        st.write(input_df)
        #st.write(f'Cardiac Risk Level: {prediction[0]}')

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: GDP Dashboard

Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
But it's otherwise a great (and did I mention _free_?) source of data.
'''

# Add some spacing
''
''
