import streamlit as st
from datetime import date
import yfinance
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')
# from fbprophet import Prophet
# from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


START="2015-01-01"
TODAY= date.today().strftime("%Y-%m-%d")
st.set_page_config(layout="wide")

st.title("Stock prediction app")

stocks = ("MUNDRAPORT","AXISBANK", "ASIANPAINT", "BAJAJ-AUTO", "BAJAJFINSV","BAJAUTOFIN","BHARTI","BPCL","BRITANNIA","CIPLA","COALINDIA","DRREDDY","EICHERMOT","GAIL","GRASIM","HCLTECH","HDFC","HDFCBANK","HEROHONDA","HINDALC0","HINDLEVER","ICICIBANK","INDUSINDBK","INFOSYSTCH","IOC","ITC","JSWSTL","KOTAKMAH","LT","MARUTI","M&M","NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE","SBIN","SHREECEM","SUNPHARMA","TELCO","TISCO","TCS","TECHM","TITAN","ULTRACEMCO","UNIPHOS","SESAGOA","WIPRO","ZEETELE")
selected_stock = st.selectbox("Select stock",stocks)

col1,col2=st.columns((1,3))

with col1:  
    n_years = st.slider("days of prediction:", 1,25)
    start = st.text_input("Enter start date","20-08-2001")
    end = st.text_input("Enter last date","20-08-2022")







b="C:\\Users\\Rapid\\Desktop\\stock\\datasets\\MinProjectMerge.CSV"
df= pd.read_csv(b)
df = df[df['Symbol'] == selected_stock]


df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')



# Filter the DataFrame by a date range
start_date = pd.to_datetime(start, format='%d-%m-%Y')
end_date = pd.to_datetime(end, format='%d-%m-%Y')
df = df[df['Date'].between(start_date, end_date)]


with col2:
    st.write(df)


#plot

fig1= go.Figure()
fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'],name="stock_close"))
fig1.layout.update(title_text="Original_data for "+selected_stock)
fig1.update_layout(xaxis_title="Date",
                  yaxis_title="Stock Price")
st.plotly_chart(fig1)


df= df[['Close']]


future_days=n_years
df['Prediction'] = df[['Close']].shift(-future_days)

x=np.array(df.drop(['Prediction'],1))[:-future_days]

y=np.array(df['Prediction'])[:-future_days]

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.25)

tree= DecisionTreeRegressor().fit(x_train,y_train)
lr= LinearRegression().fit(x_train,y_train)

x_future= df.drop(['Prediction'],1)[:-future_days]
x_future=x_future.tail(future_days)
x_future=np.array(x_future)

tree_prediction =tree.predict(x_future)

a=max(tree_prediction)
a1=np.where(tree_prediction == a)[0][0]
a1=str(a1)


lr_prediction = lr.predict(x_future)


b=max(lr_prediction)
b1=np.where(lr_prediction == b)[0][0]
b1=str(b1)

predictions= tree_prediction
valid = df[x.shape[0]:]
valid['Predictions'] = predictions

lst = list(range(1,n_years+1))




column1,column2=st.columns((1,1))

fig2= go.Figure()
fig2.add_trace(go.Scatter(x=lst, y=tree_prediction,name="stock_close"))
fig2.layout.update(title_text="tree prediction")
fig2.update_layout(xaxis_title="Days",
                  yaxis_title="Stock Price")
with column1:
    st.plotly_chart(fig2)
    st.write("According to the prediction ")
    st.write("you should buy this stock before "+a1+" days to get max output")


fig3= go.Figure()
fig3.add_trace(go.Scatter(x=lst, y=lr_prediction,name="stock_close"))
fig3.layout.update(title_text="lr prediction")
fig3.update_layout(xaxis_title="Days",
                  yaxis_title="Stock Price")
with column2:
    st.plotly_chart(fig3)
    st.write("According to the prediction ")
    st.write("you should buy this stock before "+b1+" days to get max output")
