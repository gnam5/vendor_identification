#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from textdistance import levenshtein
import unicodedata
import re
import nltk
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import string
pd.options.mode.chained_assignment = None 


# In[ ]:


df1 = pd.read_excel("Invoice data.xlsx")
df2 = pd.read_excel("API data.xlsx")
cols = ['PAN','GST']
df_1 = df1.join(df2.set_index(cols), on=cols)
df_1


# In[ ]:


df_1 = df_1[df_1['Invoice Name'].str.contains('\d+')]
df_1


# In[ ]:


df_without_numerals = df_1[~df_1['Invoice Name'].str.contains('\d+')]
df_without_numerals


# In[ ]:


df_1["Invoice Name"] = df_1["Invoice Name"].str.lower()
df_1["PAN Name"] = df_1["PAN Name"].str.lower()

data = {r'^&$' : 'and'}

df11 = df_1["Invoice Name"].str.split(' ', expand=True)
df11 = df11.replace(data, regex=True)
df_1["Invoice Name"] = df11[df11.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)

df22 = df_1["PAN Name"].str.split(' ', expand=True)
df22 = df22.replace(data, regex=True)
df_1["PAN Name"] = df22[df22.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

df_1["Invoice Name"] = df_1["Invoice Name"].apply(remove_punctuations)
df_1["PAN Name"] = df_1["PAN Name"].apply(remove_punctuations)



data = {r'^co$' : 'company', r'^comp$' : 'company',
        r'^corp$' : 'corporation', r'^corpn$' : 'corporation', r'^dba$' : 'doing business as',
        r'^inc$' : 'incorporation', r'^incorp$' : 'incorporation', r'^incorporat$' : 'incorporation', r'^incorporate$' : 'incorporation',
        r'^incorporated$' : 'incorporation', r'^intl$' : 'international', r'^intnl$' : 'international',
       r'^ltd$' : 'limited', r'^llc$' : 'limited liability company', r'^llp$' : 'limited liability partnership',
        r'^pvt$' : 'private'}



df11 = df_1["Invoice Name"].str.split(' ', expand=True)
df11 = df11.replace(data, regex=True)
df_1["Invoice Name"] = df11[df11.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)

df22 = df_1["PAN Name"].str.split(' ', expand=True)
df22 = df22.replace(data, regex=True)
df_1["PAN Name"] = df22[df22.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)

df_1


# In[ ]:


import pandas as pd
from num2words import num2words


# Define a function to convert numerals to words
def convert_numerals_to_words(text):
    words = []
    for word in text.split():
        if word.isdigit():
            words.append(num2words(int(word)))
        else:
            words.append(word)
    return ' '.join(words)

# Apply the function to the 'text' column
df_1["Invoice Name_1"] = df_1["Invoice Name"].apply(convert_numerals_to_words)

# Print the updated dataframe
df_1


# In[ ]:


from name_matching.name_matcher import NameMatcher

from thefuzz import fuzz

matcher = NameMatcher()



matcher.set_distance_metrics([ 'discounted_levenshtein','fuzzy_wuzzy_token_sort'])

matcher.load_and_process_master_data('Invoice Name_1', df_1)
df = matcher.match_names(to_be_matched=df_1, column_matching='PAN Name')
df

