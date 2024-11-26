# In[ ]:

import numpy as np
import pandas as pd



# In[ ]:

fake_data=pd.read_csv('Fake.csv')



# In[ ]:

true_data=pd.read_csv('True.csv')



# In[ ]:

fake_data.head(10)



# In[ ]:

true_data.head(10)



# In[ ]:

fake_data['subject'].value_counts()



# In[ ]:

true_data['subject'].value_counts()



# In[ ]:

fake_data['category']=1



# In[ ]:

true_data['category']=0



# In[ ]:

fake_data.head()



# In[ ]:

df=pd.concat([fake_data,true_data]).reset_index(drop=True)



# In[ ]:

df.head()



# In[ ]:

df.tail(20)



# In[ ]:

#Data visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline



# In[ ]:

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,7))
#df=pd.concat([fake_data,true_data]).reset_index(drop=True)
sns.countplot(df["category"])
plt.legend()



# In[ ]:

plt.figure(figsize=(10,7))
sns.countplot(df["subject"])
plt.legend()



# In[ ]:

# Data Cleaning 

# df.df[["text","category"]]
df.drop(columns=["title", "subject","date"], inplace=True)




# In[ ]:

df.head()



# In[ ]:

df.isna().sum()*100/len(df)



# In[ ]:

blanks=[]  #checking if there is empty string in TEXT column

for index,text in df["text"].items():
    if text.isspace():
        blanks.append(index)

len(blanks)



# In[ ]:

blanks



# In[ ]:

df["text"][10922]



# In[ ]:

df["text"][10923]



# In[ ]:

df.shape 



# In[ ]:

df.drop(blanks,inplace=True)



# In[ ]:

df.shape



# In[ ]:

#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer
#import spacy 
#import re

#nlp=spacy.load("en_core_web_sm")
#



# In[ ]:

!pip install spacy



# In[ ]:

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy 
import re

#nlp=spacy.load("en_core_web_sm")



# In[ ]:

!python -m spacy download en_core_web_sm



# In[ ]:

nlp=spacy.load("en_core_web_sm")



# In[ ]:

lemma=WordNetLemmatizer()




# In[ ]:

#stopwords of spacy
list1=nlp.Defaults.stop_words
print(len(list1))

#stopwords of NLTK
list2=stopwords.words('english')
print(len(list2))

#combining the stopword list
Stopwords=set((set(list1)|set(list2)))
print(len(Stopwords))



# In[ ]:

#Cleaning the text
def clean_text(text):
    
    """
    It takes text as an input and clean it by applying several methods
    
    """
    
    string = ""
    
    #lower casing
    text=text.lower()
    
    #simplifying text
    text=re.sub(r"i'm","i am",text)
    text=re.sub(r"he's","he is",text)
    text=re.sub(r"she's","she is",text)
    text=re.sub(r"that's","that is",text)
    text=re.sub(r"what's","what is",text)
    text=re.sub(r"where's","where is",text)
    text=re.sub(r"\'ll"," will",text)
    text=re.sub(r"\'ve"," have",text)
    text=re.sub(r"\'re"," are",text)
    text=re.sub(r"\'d"," would",text)
    text=re.sub(r"won't","will not",text)
    text=re.sub(r"can't","cannot",text)
    
    #removing any special character
    text=re.sub(r"[-()\"#!@$%^&*{}?.,:]"," ",text)
    text=re.sub(r"\s+"," ",text)
    text=re.sub('[^A-Za-z0-9]+',' ', text)
    
    for word in text.split():
        if word not in Stopwords:
            string+=lemma.lemmatize(word)+" "
    
    return string



# In[ ]:

df["text"][10]
clean_text(df["text"][10])



# In[ ]:

#cleaning the whole data
df["text"] = df["text"].apply(clean_text)




# In[ ]:

df["text"][0]



# In[ ]:

!pip install wordcloud



# In[ ]:

from wordcloud import WordCloud



# In[ ]:

#True News
plt.figure(figsize=(20,20))
Wc=WordCloud(max_words=500,width=1600,height=800).generate(" ".join(df[df.category==0].text))
plt.axis("off")
plt.imshow(Wc,interpolation='bilinear')



# In[ ]:

from PTL import Image


