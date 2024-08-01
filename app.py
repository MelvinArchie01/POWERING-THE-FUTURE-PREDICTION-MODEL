import streamlit as st
import pandas as pd




df = pd.read_csv("WorldBank Renewable Energy Consumption_WorldBank Renewable Energy Consumption.csv")
# app title
st.title('Powering the Future: A Data-Driven Approach to Renewable Energy Forecasting')

#creating a paragraph

st.write('''  Renewable Energy Consumption has varied over the years with countries shifting to other sources of energy
         As the world transitions to a low-carbon economy, 
         reliable forecasts of renewable energy consumption are essential.
          By delving into nearly three decades of data,
          we aim to develop a robust predictive model that can inform energy planning,
          investment, and policymaking. Our work is a step towards a more sustainable
          and resilient energy future.
                                            ''')

 
st.write(df.head(5)) #printing the first 5 rows


#having user slider

num_rows = st.slider("Select the number of rows", min_value = 1, max_value = len(df), value = 5)
st.write("Here are the rows you have selected in the Dataset")
st.write(df.head(num_rows)) #st.write is the print function in python
st.write('The number of rows and columns in the dataset')
st.write(df.shape)
st.write("number of duplicates:", df[df.duplicated()])

#------------------------------------------------------------------------------------------------------------
if st.checkbox('check for duplicates'):
   st.write(df[df.duplicated()])

if st.checkbox('total number of duplicates'):
   st.write(df.duplicated().sum())



#changing the dtype from string to integers

df['Year'] = pd.to_datetime(df['Year'])
df['Year'] = df['Year'].dt.year
def clean_outliers(column):
  mean = df[column].mean()
  std = df[column].std()
  threshold = 3
  lower_limit = mean - (threshold * std)
  upper_limit = mean + (threshold * std)

  return df[(df[column]>=lower_limit) & (df[column]<=upper_limit)]

columns = ['Year', 'Energy Consump.']
for column in columns:
  new_df = clean_outliers(column)

  # Drop 'Country Code' column
new_df = df.drop('Country Code', axis=1)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# Display the dataframe
st.write("The new Dataset without the category code:",new_df.head(3))

# Split the data into features and target
X = new_df.drop('Energy Consump.', axis = 1)
y = new_df['Energy Consump.']


####################################################################################################
# Encode categorical variables
encoded_columns = ['Country Name', 'Income Group', 'Indicator Code', 'Indicator Name', 'Region']
le_dict = {col: LabelEncoder() for col in encoded_columns}

for column in encoded_columns:
    le_dict[column].fit(new_df[column])
    new_df[column] = le_dict[column].transform(new_df[column])



#Training the Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
from sklearn.model_selection import train_test_split
X = new_df.drop('Energy Consump.', axis = 1)
y = new_df['Energy Consump.']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
r2 = r2_score(y_test, y_pred)
st.write("R-squared:", r2)
mae = mean_absolute_error(y_test, y_pred)
st.write("Mean Absolute Error:", mae)

mse = mean_squared_error(y_test, y_pred)
st.write("Mean Squared Error:", mse)





st.sidebar.write("## Enter new data for prediction")

Country_Name = st.sidebar.selectbox("Country Name", le_dict['Country Name'].classes_)

Indicator_Code = st.sidebar.selectbox("Indicator Code", le_dict['Indicator Code'].classes_)
Indicator_Name = st.sidebar.selectbox("Indicator Name", le_dict['Indicator Name'].classes_)
Region= st.sidebar.selectbox("Region", le_dict['Region'].classes_)
Income_Group= st.sidebar.selectbox("Income Group", le_dict['Income Group'].classes_)
Year = st.sidebar.number_input("Year")
# Encode user input
encoded_input = [
    le_dict['Country Name'].transform([Country_Name])[0],
   
    le_dict['Indicator Code'].transform([Indicator_Code])[0],
    le_dict['Indicator Name'].transform([Indicator_Name])[0],
    le_dict['Region'].transform([Region])[0],
    le_dict['Income Group'].transform([Income_Group])[0],
    Year
]

# Predict using the model
if st.sidebar.button('Renewable Energy Consumption'):
    prediction = model.predict([encoded_input])[0]
    st.sidebar.write('Predicted Energy Consumption:', prediction)















    




















