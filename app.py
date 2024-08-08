import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# App title 
st.title("RENEWABLE ENERGY ANALYSIS")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Homepage", "Data Information", "Visualization", "Machine Learning Model"])

# Load the dataset
df = pd.read_csv("Renewable Energy.csv")

# Ensure 'Country Code' is not used in the model (assuming it wasn't dropped earlier)
if 'Country Code' in df.columns:
    df = df.drop('Country Code', axis=1)

# Convert 'Year' column to integers (assuming the format is "day/month/year")
df['Year'] = pd.to_datetime(df['Year'], format='%d/%m/%Y').dt.year


# Homepage
if option == "Homepage":
    st.write('''
             
    ### Welcome to the Renewable Energy Analysis Web Application!
    
    What Is Renewable Energy?
    
    Renewable energy originates from inexhaustible, naturally replenished sources such as the sun, wind, and tides.
    It plays a crucial role in generating electricity, heating and cooling spaces and water, and powering transportation.
    
    **Instructions for Using the Web Application:**
    
    The aim of this web application is to predict energy consumption using various parameters such as region, country name, year, indicator name, indicator code, and income group.
    ''')

# Data Information
elif option == "Data Information":
    st.write("### Data Information")
    
    # Display the first 5 rows
    st.write('The first five rows of the dataset:', df.head())
    
    # User input: number of rows to display
    num_rows = st.slider("Select the number of rows", min_value=1, max_value=len(df), value=5)
    st.write("Here are the rows you have selected from the dataset:")
    st.write(df.head(num_rows))
    
    # Display the shape of the dataset
    st.write('Number of rows and columns in the dataset:', df.shape)
    
    # Display the summary statistics of the dataset
    st.write('Summary statistics of the dataset:')
    st.write(df.describe())
    
    # Check for duplicates
    if st.checkbox("Check for duplicates"):
        st.write(df[df.duplicated()])
    
    # Total number of duplicates
    if st.checkbox("Check for total number of duplicates"):
        st.write(df.duplicated().sum())

# Visualization
elif option == "Visualization":
    st.write("### Data Visualization")
    
    # Plot histogram of energy consumption
    st.write('Renewable energy distribution')
    sns.histplot(df['Energy Consump.'], kde=True)
    plt.xlabel('Energy Consump.')
    plt.ylabel('Frequency')
    plt.title('Histogram of Energy Consump.')
    st.pyplot(plt.gcf())

# Machine Learning Model
elif option == "Machine Learning Model":
    st.write("### Machine Learning Model")

    st.title('Energy Consumption Prediction')

    # Encode categorical variables
    encoded_columns = ['Region', 'Income Group', 'Indicator Name', 'Country Name', 'Indicator Code']
    le_dict = {col: LabelEncoder() for col in encoded_columns}

    for column in encoded_columns:
        df[column] = le_dict[column].fit_transform(df[column])

    # Prepare features and target variable
    X = df.drop('Energy Consump.', axis=1)
    y = df['Energy Consump.']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Initialize the Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    rf_regressor.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_regressor.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"R-squared: {r2:.4f}")
    
    # User input for new data
    st.write("## Enter new data for prediction")
    Country_Name = st.selectbox("Country Name", le_dict['Country Name'].classes_)
    Income_Group = st.selectbox("Income Group", le_dict['Income Group'].classes_)
    Indicator_Code = st.selectbox("Indicator Code", le_dict['Indicator Code'].classes_)
    Indicator_Name = st.selectbox("Indicator Name", le_dict['Indicator Name'].classes_)
    Region = st.selectbox("Region", le_dict['Region'].classes_)
    Year = st.number_input("Year", min_value=int(df['Year'].min()), max_value=2050, step=1)
    
    # Encode user input
    encoded_input = {
        'Country Name': le_dict['Country Name'].transform([Country_Name])[0],
        'Income Group': le_dict['Income Group'].transform([Income_Group])[0],
        'Indicator Code': le_dict['Indicator Code'].transform([Indicator_Code])[0],
        'Indicator Name': le_dict['Indicator Name'].transform([Indicator_Name])[0],
        'Region': le_dict['Region'].transform([Region])[0],
        'Year': Year
    }
    
    # Convert to DataFrame and standardize
    encoded_input_df = pd.DataFrame([encoded_input])
    encoded_input_scaled = scaler.transform(encoded_input_df)
    
    # Predict using the model
    if st.button('Predict Energy Consump.'):
        prediction = rf_regressor.predict(encoded_input_scaled)[0]
        st.write(f'Predicted energy consumption: {prediction:.4f}')

        if prediction > 100:
            st.write("The energy consumption is EXCELLENT.")
        elif 50 < prediction <= 100:
            st.write("The energy consumption is GOOD.")
        else:
            st.write("The energy consumption is AVERAGE.")
