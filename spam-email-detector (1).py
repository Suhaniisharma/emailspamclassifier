#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import sys
import sklearn
df = pd.read_csv('spam.csv', encoding='latin1')
df.sample(5)
df.shape


# # 1. Data Cleaning

# In[5]:


df.info()


# In[6]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[7]:


df.sample(5)


# In[8]:


df.rename(columns={'v1':'spam/ham','v2':'Text_Message'},inplace=True)


# In[9]:


df.sample(5)


# In[10]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[11]:


df['spam/ham']=encoder.fit_transform(df['spam/ham'])


# In[12]:


df.head()


# In[13]:


#missing values
df.isnull().sum()


# In[14]:


#duplicate value
df.duplicated().sum()


# In[15]:


df=df.drop_duplicates(keep='first')


# In[16]:


df.duplicated().sum()


# # 2.EDA

# In[17]:


import matplotlib.pyplot as plt
plt.pie(df['spam/ham'].value_counts(),labels=['Ham','Spam'],autopct="%0.2f")
plt.show


# In[18]:


pip install ntlk


# In[19]:


import nltk


# In[20]:


nltk.download('punkt')


# In[21]:


#Number of characters in msg
df['numberOfChar']=df['Text_Message'].apply(len)


# In[22]:


df.head()


# In[23]:


#Number of words
df['NumberOfWords']=df['Text_Message'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[24]:


df.head()


# In[25]:


#Number of sentences
df['NumberOfSent']=df['Text_Message'].apply(lambda x:len(nltk.sent_tokenize(x)))
df.head()


# # 3.Data preprocessing
# #Lowercase
# #Tokenization
# #Removing special characters
# #Removing stop words and punctutation
# #stemming

# In[26]:


import nltk
from nltk.corpus import stopwords #stopwords
import string #punctuations

nltk.download('stopwords')


# In[27]:


def transform_data(Text_Message):
    Text_Message=Text_Message.lower()#lowercase
    Text_Message=nltk.word_tokenize(Text_Message)#tokenization
    
    #Remove special characters
    x=[]
    for i in Text_Message:
        if i.isalnum():
            x.append(i)
    Text_Message=x[:]
    x.clear()
    
    #Removing stopwords $ punctutation
    for i in Text_Message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            x.append(i)
            
    Text_Message=x[:]
    x.clear()
    
    #stemming
    for i in Text_Message:
        x.append(ps.stem(i))
    return " ".join(x)


# In[30]:


transform_data('Hello how are you SAkshi??? @@@@@@@@@@ my hobby is Dancing')


# In[29]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem('Sleeping')


# In[31]:


df['transformData']=df['Text_Message'].apply(transform_data)


# In[32]:


df.head()


# In[52]:


#creating word pool to count imp wordfrom spam and ham
get_ipython().system('pip install wordcloud')


# In[33]:


from wordcloud import WordCloud
wc=WordCloud(width=500,height=500,min_font_size=10,background_color='black')


# In[34]:


spam_wc=wc.generate(df[df['spam/ham']==1]['transformData'].str.cat(sep=" "))


# In[35]:


plt.imshow(spam_wc)


# In[9]:


ham_wc=wc.generate(df[df['spam/ham']==0]['transformData'].str.cat(sep=" "))


# In[36]:


plt.imshow(ham_wc)


# In[53]:


spam_corpus=[]
for msg in df[df['spam/ham']==1]['transformData'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[54]:


len(spam_corpus)


# In[55]:


import seaborn as sns


# In[43]:


from collections import Counter
Counter(spam_corpus).most_common(30)


# # 4.Model Building

# In[7]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[8]:


X = tfidf.fit_transform(df['transformData']).toarray()


# In[46]:


X.shape


# In[47]:


y = df['spam/ham'].values


# In[48]:


y


# In[56]:


from sklearn.model_selection import train_test_split


# In[3]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[2]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[59]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[1]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[40]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[41]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[56]:


#tfidf->mnb


# # Model improvement

# In[186]:





# In[60]:


import pickle

# Assuming tfidf and mnb are defined elsewhere

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('model.pkl','wb') as f:
    pickle.dump(mnb, f, protocol=pickle.HIGHEST_PROTOCOL)


# In[61]:


import sys
print(sys.executable)


# In[62]:


print("numpy",np.__version__)
print("pandas",pd.__version__)
print("sklearn",sklearn.__version__)
print("nltk",nltk.__version__)


# In[ ]:




