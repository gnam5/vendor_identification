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
df_inovice_gst = df1.join(df2.set_index(cols), on=cols)
df_inovice_gst.head()


# In[ ]:


df_inovice_gst["Company Name"] = df_inovice_gst["Company Name"].str.lower()
df_inovice_gst["Legal Name"] = df_inovice_gst["Legal Name"].str.lower()
df_inovice_gst["Trade Name"] = df_inovice_gst["Trade Name"].str.lower()

data = {r'^&$' : 'and'}
            
df11 = df_inovice_gst["Company Name"].str.split(' ', expand=True)
df11 = df11.replace(data, regex=True)
df_inovice_gst["Company Name"] = df11[df11.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
            
df22 = df_inovice_gst["Legal Name"].str.split(' ', expand=True)
df22 = df22.replace(data, regex=True)
df_inovice_gst["Legal Name"] = df22[df22.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
 
df33 = df_inovice_gst["Trade Name"].str.split(' ', expand=True)
df33 = df33.replace(data, regex=True)
df_inovice_gst["Trade Name"] = df22[df22.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

df_inovice_gst["Company Name"] = df_inovice_gst["Company Name"].apply(remove_punctuations)
df_inovice_gst["Legal Name"] = df_inovice_gst["Legal Name"].apply(remove_punctuations)
df_inovice_gst["Trade Name"] = df_inovice_gst["Trade Name"].apply(remove_punctuations)

            
data = {r'^co$' : 'company', r'^comp$' : 'company',
                    r'^corp$' : 'corporation', r'^corpn$' : 'corporation', r'^dba$' : 'doing business as',
                    r'^inc$' : 'incorporation', r'^incorp$' : 'incorporation', r'^incorporat$' : 'incorporation', r'^incorporate$' :                            'incorporation',
                    r'^incorporated$' : 'incorporation', r'^intl$' : 'international', r'^intnl$' : 'international',
                    r'^ltd$' : 'limited', r'^llc$' : 'limited liability company', r'^llp$' : 'limited liability partnership',
                    r'^pvt$' : 'private'
                }

df11 = df_inovice_gst["Company Name"].str.split(' ', expand=True)
df11 = df11.replace(data, regex=True)
df_inovice_gst["Company Name"] = df11[df11.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
            
df22 = df_inovice_gst["Legal Name"].str.split(' ', expand=True)
df22 = df22.replace(data, regex=True)
df_inovice_gst["Legal Name"] = df22[df22.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
 
df33 = df_inovice_gst["Trade Name"].str.split(' ', expand=True)
df33 = df33.replace(data, regex=True)
df_inovice_gst["Trade Name"] = df22[df22.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)
df_inovice_gst.head()


# In[ ]:


from name_matching.name_matcher import NameMatcher

from thefuzz import fuzz

matcher = NameMatcher()

matcher.set_distance_metrics([ 'discounted_levenshtein', 'fuzzy_wuzzy_token_sort'])
matcher.load_and_process_master_data('Company Name', df_inovice_gst)
df = matcher.match_names(to_be_matched=df_inovice_gst, column_matching='Legal Name')
            

#df.rename(columns = {'original_name':'Company Name','match_name':'Legal Name'}, inplace = True)
            
df.loc[df['score'] == 100, 'Alert_1'] = 1  
df.loc[df['score'] != 100, 'Alert_1'] = 0
df.loc[df['score'] == 100, 'Name Alert'] = 'Correct' 
df.loc[df['score'] != 100, 'Name Alert'] = 'Invoice Name differs from PAN Registered Name'
df_final_1 = df.drop(['score','match_index','original_name','match_name'], axis=1)
df_final_1


# In[ ]:


df_final_1 = pd.concat([df_inovice_gst, df_final_1],axis=1)
df_final_1


# In[ ]:


df_correct = df_final_1.loc[df_final_1['Alert_1'] == 1]
df_incorrect = df_final_1.loc[df_final_1['Alert_1'] == 0]
df_incorrect = df_incorrect.drop(['Alert_1','Name Alert'], axis=1)
df_incorrect


# In[ ]:


matcher.set_distance_metrics([ 'discounted_levenshtein', 'fuzzy_wuzzy_token_sort'])
matcher.load_and_process_master_data('Company Name', df_incorrect)
df = matcher.match_names(to_be_matched=df_incorrect, column_matching='Trade Name')

df.loc[df['score'] == 100, 'Alert_1'] = 1  
df.loc[df['score'] != 100, 'Alert_1'] = 0
df.loc[df['score'] == 100, 'Name Alert'] = 'Correct' 
df.loc[df['score'] != 100, 'Name Alert'] = 'Invoice Name differs from GST Registered Name'
                    
df_name_matching_2 = df.drop(['score','match_index','original_name','match_name'], axis=1)
df_name_matching_2


# In[ ]:


df_final_2 = pd.concat([df_incorrect, df_name_matching_2],axis=1)
df_final_2


# In[ ]:


df = pd.concat([df_correct, df_final_2],axis=0)
df


# In[ ]:


df.loc[(df['Status'] == "Active"), 'Alert_2'] = 1  
df.loc[(df['Status'] != "Active"), 'Alert_2'] = 0
            
df.loc[(df_inovice_gst['Status'] == "Active"), 'Status Alert'] = 'Correct'
df.loc[(df_inovice_gst['Status'] != "Active"), 'Status Alert'] = 'GST is Canceled' 
df         


# In[ ]:


conditions = [(df['Alert_1'] == 1) & (df['Alert_2'] == 1),
              (df['Alert_1'] == 1) & (df['Alert_2'] == 0),
            (df['Alert_1'] == 0) & (df['Alert_2'] == 1),
            (df['Alert_1'] == 0) & (df['Alert_2'] == 0)]


values = [1, 1, 1, 0]
df['Alert'] = np.select(conditions, values)
df


# In[ ]:



final = df.drop(['Alert_1','Alert_2'], axis=1)
final


# In[ ]:


df.loc[(df['Status'] == 'Active'), 'Alert'] = 1  
df.loc[(df['Status'] != 'Active'), 'Alert'] = 0
df.head()


# In[ ]:


df_1 = df.loc[df['Alert'] == 1]
df_2 = df.loc[df['Alert'] == 0]
df_1 = df_1.drop(['Alert'], axis=1)
df_2


# In[ ]:


df_1["Invoice Name"] = df_1["Invoice Name"].str.lower()
df_1["Leagal Name"] = df_1["Leagal Name"].str.lower()
df_1["Trade Name"] = df_1["Trade Name"].str.lower()


# In[ ]:


data = {r'^&$' : 'and'}

df11 = df_1["Invoice Name"].str.split(' ', expand=True)
df11 = df11.replace(data, regex=True)
df_1["Invoice Name"] = df11[df11.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)

df22 = df_1["Leagal Name"].str.split(' ', expand=True)
df22 = df22.replace(data, regex=True)
df_1["Leagal Name"] = df22[df22.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)


df33 = df_1["Trade Name"].str.split(' ', expand=True)
df33 = df33.replace(data, regex=True)
df_1["Trade Name"] = df33[df33.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)


# In[ ]:


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

df_1["Invoice Name"] = df_1["Invoice Name"].apply(remove_punctuations)
df_1["Leagal Name"] = df_1["Leagal Name"].apply(remove_punctuations)
df_1["Trade Name"] = df_1["Trade Name"].apply(remove_punctuations)


# In[ ]:


data = {r'^co$' : 'company', r'^comp$' : 'company',
        r'^corp$' : 'corporation', r'^corpn$' : 'corporation', r'^dba$' : 'doing business as',
        r'^inc$' : 'incorporation', r'^incorp$' : 'incorporation', r'^incorporat$' : 'incorporation', r'^incorporate$' : 'incorporation',
        r'^incorporated$' : 'incorporation', r'^intl$' : 'international', r'^intnl$' : 'international',
       r'^ltd$' : 'limited', r'^llc$' : 'limited liability company', r'^llp$' : 'limited liability partnership',
        r'^pvt$' : 'private'}


# In[ ]:


df11 = df_1["Invoice Name"].str.split(' ', expand=True)
df11 = df11.replace(data, regex=True)
df_1["Invoice Name"] = df11[df11.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)

df22 = df_1["Leagal Name"].str.split(' ', expand=True)
df22 = df22.replace(data, regex=True)
df_1["Leagal Name"] = df22[df22.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)


df33 = df_1["Trade Name"].str.split(' ', expand=True)
df33 = df33.replace(data, regex=True)
df_1["Trade Name"] = df33[df33.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)),axis=1)


# In[ ]:


from name_matching.name_matcher import NameMatcher

from thefuzz import fuzz

matcher = NameMatcher()


import abydos.distance as abd
matcher.set_distance_metrics([ 'discounted_levenshtein','fuzzy_wuzzy_token_sort'])

matcher.load_and_process_master_data('Invoice Name', df_1)
df = matcher.match_names(to_be_matched=df_1, column_matching="Leagal Name")
df


# In[ ]:


df.loc[df['score'] == 100, 'Alert'] = 1 
df.loc[df['score'] != 100, 'Alert'] = 0
df33 = df.drop(['score','match_index','original_name','match_name'], axis=1)
df33


# In[ ]:


df_final = pd.concat([df_1, df33],axis=1)
df_final


# In[ ]:


df_3 = df_final.loc[df_final['Alert'] == 1]
df_4 = df_final.loc[df_final['Alert'] == 0]
df_4 = df_4.drop(['Alert'], axis=1)
df_4


# In[ ]:


from name_matching.name_matcher import NameMatcher

from thefuzz import fuzz

matcher = NameMatcher()


import abydos.distance as abd
matcher.set_distance_metrics([ 'discounted_levenshtein','fuzzy_wuzzy_token_sort'])

matcher.load_and_process_master_data('Invoice Name', df_4)
df = matcher.match_names(to_be_matched=df_4, column_matching="Trade Name")
df


# In[ ]:


df.loc[df['score'] == 100, 'Alert'] = 1 
df.loc[df['score'] != 100, 'Alert'] = 0
df44 = df.drop(['score','match_index','original_name','match_name'], axis=1)
df44


# In[ ]:


final1 = pd.concat([df_4, df44],axis=1)
final1


# In[ ]:


final2 = pd.concat([final1, df_3],axis=0)
final2


# In[ ]:


final3 = pd.concat([final2, df_2],axis=0)
final3

