import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# App title 
st.title("RENEWABLE ENERGY ANALYSIS")

# Create a paragraph  
st.write('''
What Is Renewable Energy?

    Renewable energy originates from inexhaustible, naturally replenished sources as,
          the sun, wind, and tides.
          It plays a crucial role in generating electricity, 
         heating and cooling spaces and water, and powering transportation. 

Instructions for Using the Web Application:

  The aim of this web application is to predict energy consumption using various parameters such as region,
          country name, year, indicator name, 
         indicator code, and income group. 
         To get started, please use the options on the left to explore and input the relevant data. 
       
''')

# Load the dataset
df = pd.read_csv("Renewable Energy.csv")

# Display the first 5 rows
st.write('The first five rows', df.head())

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
if st.checkbox("Check for total number duplicates"):
    st.write(df.duplicated().sum())

# Drop 'Country Code' column
df = df.drop('Country Code', axis=1)

# Convert 'Year' column to integers (assuming the format is "day/month/year")
df['Year'] = pd.to_datetime(df['Year'], format='%d/%m/%Y').dt.year

# Encode categorical variables
encoded_columns = ['Region', 'Income Group', 'Indicator Name', 'Country Name', 'Indicator Code']
le_dict = {col: LabelEncoder() for col in encoded_columns}

for column in encoded_columns:
    le_dict[column].fit(df[column])
    df[column] = le_dict[column].transform(df[column])

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
st.sidebar.write("## Enter new data for prediction")
Country_Name = st.sidebar.selectbox("Country Name", le_dict['Country Name'].classes_)
Income_Group = st.sidebar.selectbox("Income Group", le_dict['Income Group'].classes_)
Indicator_Code = st.sidebar.selectbox("Indicator Code", le_dict['Indicator Code'].classes_)
Indicator_Name = st.sidebar.selectbox("Indicator Name", le_dict['Indicator Name'].classes_)
Region = st.sidebar.selectbox("Region", le_dict['Region'].classes_)
Year = st.sidebar.number_input("Year", min_value=int(df['Year'].min()), max_value=2050, step=1)

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
if st.sidebar.button('Predict Energy Consump.'):
    prediction = rf_regressor.predict(encoded_input_scaled)[0]
    st.sidebar.write('Predicted energy consumption:', prediction)


