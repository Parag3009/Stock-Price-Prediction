import streamlit as st
from datetime import date
import base64
import openai
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go


openai.api_key = "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"

st.set_page_config(layout="wide")

st.title("Stock Prediction App")

stocks = ["AAPL", "GOOG", "MSFT", "GME"]
selected_stock = st.selectbox("Select Stock", stocks)

col1, col2 = st.columns((1, 3))

with col1:
    n_years = st.slider("Days of Prediction", 1, 25)
    start = st.text_input("Enter Start Date (YYYY-MM-DD)", "2021-01-01")
    end = st.text_input("Enter End Date (YYYY-MM-DD)", str(date.today()))

symbol = selected_stock
start_date = start
end_date = end

# Download data from Yahoo Finance API
data = yf.download(symbol, start=start_date, end=end_date)
df = pd.DataFrame(data)
df.sort_values(by=["Date"], inplace=True, ascending=True)

# Reset index
df.reset_index(inplace=True)

with col2:
    # Display the last 10 rows of data
    st.write(df.tail(10))

# Plot the stock price data
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Stock Price"))
fig1.layout.update(title_text="Original Data for " + selected_stock)
fig1.update_layout(xaxis_title="Date", yaxis_title="Stock Price")
st.plotly_chart(fig1)

# Prepare data for prediction
df = df[["Close"]]
future_days = n_years
df["Prediction"] = df[["Close"]].shift(-future_days)

x = np.array(df.drop(["Prediction"], 1))[:-future_days]
y = np.array(df["Prediction"])[:-future_days]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Train models
tree = DecisionTreeRegressor().fit(x_train, y_train)
lr = LinearRegression().fit(x_train, y_train)

# Predict stock price
x_future = df.drop(["Prediction"], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)

tree_prediction = tree.predict(x_future)
a = max(tree_prediction)
a1 = np.where(tree_prediction == a)[0][0]

lr_prediction = lr.predict(x_future)
b = max(lr_prediction)
b1 = np.where(lr_prediction == b)[0][0]

# Display predictions
predictions = tree_prediction
valid = df[x.shape[0]:]
valid["Predictions"] = predictions

lst = list(range(1, n_years + 1))

with st.expander("View Predictions"):
    col3, col4 = st.columns(2)
    
        # Plot the decision tree model prediction
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=lst, y=tree_prediction, name="Stock Price"))
    fig2.layout.update(title_text="Decision Tree Prediction")
    fig2.update_layout(xaxis_title="Days", yaxis_title="Stock Price")
    st.plotly_chart(fig2)
    st.write(
        f"If you want to get maximum output, you should buy this stock before {a1} days."
    )
with st.expander("View 2nd Prediction"):
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=lst, y=lr_prediction, name="Stock Price"))
    fig2.layout.update(title_text="linear Prediction")
    fig2.update_layout(xaxis_title="Days", yaxis_title="Stock Price")
    st.plotly_chart(fig2)
    st.write(
        f"If you want to get maximum output, you should buy this stock before {b1} days."
    )


#  completion = openai.Completion.create(
#             engine=model_engine,
#             prompt=prom,
#             max_tokens=1024,
#             n=1,
#             stop=None,
#             temperature=0.5,
#         )

#         response = completion.choices[0].text
#         # print(response)
#         st.write(response)
model_engine = "text-davinci-003"
inp=st.text_input("Enter your doubts here")
m=st.button("Search",key="3")
if m:
    prom=inp+" in 500 words"
    

    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prom,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    response = completion.choices[0].text
    # print(response)
    st.write(response)
    st.write("")
    st.write("")
    st.write("")

promp="history of "+selected_stock+ " in yfinance in 3000 words"
completion1 = openai.Completion.create(
        engine=model_engine,
        prompt=promp,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
st.title("History of "+selected_stock+" :")
response1 = completion1.choices[0].text
# print(response)
st.write(response1)


st.write("")
st.write("")
st.write("")
st.write("")
promp="should we invest in "+selected_stock+ "  in 1000 words"
completion1 = openai.Completion.create(
        engine=model_engine,
        prompt=promp,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
st.title("Investment in "+selected_stock+" :")
response2 = completion1.choices[0].text
# print(response)
st.write(response2)
