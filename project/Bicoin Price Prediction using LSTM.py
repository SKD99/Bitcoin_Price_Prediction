#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense,LSTM,Dropout

from sklearn.preprocessing import  MinMaxScaler


# In[2]:


data_dir='BTC_USD.csv'
df= pd.read_csv(r"C:\Users\DELL\Dropbox\My PC (DESKTOP-S92PGJG)\Desktop\project\BTC_USD.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


print(df.columns)


# In[8]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))

df[r'Open'].plot()

df[r'Close'].plot()
df[r'Volume_(BTC)'].plot()
df[r'Volume_(Currency)'].plot()
df[r'Weighted_Price'].plot()
plt.ylabel(None)
plt.xlabel(None)
plt.title('Opening and Closing Price of Bitcoin')
plt.legend(['Open Price','Close Price'])
plt.tight_layout()
plt.show()


# In[9]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))

df[r'High'].plot()
df[r'Low'].plot()
plt.ylabel(None)
plt.xlabel(None)
plt.title('High and Low Price of Bitcoin')
plt.legend(['High Price','Low Price'])
plt.tight_layout()
plt.show()


# In[10]:


n_cols=1
dataset=df[r'Close']
dataset=pd.DataFrame(dataset)
data=dataset.values

data.shape


# In[11]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(np.array(data))


# In[12]:


train_size=int(len(data)*0.75)
test_size=len(data)-train_size
print('Train Size:',train_size,'Test Size:',test_size)


# In[13]:


train_data=scaled_data[0:train_size,:]
train_data.shape


# In[14]:


x_train=[]
y_train=[]
time_steps=60
n_cols=1
for i in range(time_steps,len(train_data)):
    x_train.append(train_data[i-time_steps:i,:n_cols])
    y_train.append(train_data[i,:n_cols])
    if i<= time_steps:
        print('X_train:',x_train)
        print('Y_train:',y_train)


# In[15]:


x_train,y_train=np.array(x_train),np.array(y_train)


# In[16]:


x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],n_cols))


# In[17]:


x_train.shape,y_train.shape


# In[18]:


model=Sequential([
    LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],n_cols)),
    LSTM(64,return_sequences=False),
    Dense(32),
    Dense(16),
    Dense(n_cols)
    ])
model.compile(optimizer='adam',loss='mse',metrics='mean_absolute_error')


# In[19]:


model.summary()


# In[32]:


history=model.fit(x_train,y_train,)


# In[33]:


plt.figure(figsize=(12,8))
plt.plot(history.history["loss"])
plt.plot(history.history["mean_absolute_error"])
plt.legend(['Mean Squared Error','Mean Absolute Error'])
plt.title("Losses")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


# In[34]:


time_steps=60
test_data=scaled_data[train_size-time_steps:,:]
x_test=[]
y_test=[]
n_cols=1

for i in range(time_steps,len(test_data)):
    x_test.append(test_data[i-time_steps:i,0:n_cols])
    y_test.append(test_data[i,0:n_cols])
    
x_test,y_test=np.array(x_test),np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], n_cols))


# In[35]:


x_test.shape,y_test.shape


# In[24]:


predictions=model.predict(x_test)


# In[36]:


predictions=scaler.inverse_transform(predictions)
predictions.shape


# In[37]:


y_test=scaler.inverse_transform(y_test)
rmse=np.sqrt(np.mean(y_test-predictions)**2).round(2)


# In[49]:


preds_acts=pd.DataFrame(columns=['Predictions','Actuals'])
preds_acts['Predictions']=predictions.flatten()
preds_acts['Actuals']=y_test.flatten()



# In[50]:


preds_acts.head()


# In[51]:


print(preds_acts.columns)


# In[52]:


# Example: Check if 'Predictions' column is present (case-sensitive)
print('Predictions' in preds_acts.columns)


# In[53]:


plt.figure(figsize=(16,6))
plt.plot(preds_acts['Predictions'])
plt.plot(preds_acts['Actuals'])
plt.legend(['Predictions','Actuals'])
plt.show()


# In[55]:


train=dataset.iloc[:train_size,0:1]
test=dataset.iloc[train_size:,0:1]
test['Predictions']=predictions
plt.figure(figsize=(16,6))
plt.title('Bitcoin Stock Price Predictions',fontsize=20)
plt.xlabel('Timestamp',fontsize=18)
plt.ylabel('Close ',fontsize=18)
plt.plot(train[r'Close'],linewidth=3)
plt.plot(test[r'Close'],linewidth=3)

plt.plot(test['Predictions'],linewidth=3)
plt.legend(['Train','Test','Predictions'])    

