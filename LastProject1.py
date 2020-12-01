#!/usr/bin/env python
# coding: utf-8

# In[27]:


get_ipython().system('pip install beautifulsoup4')
get_ipython().system('pip install lxml')
get_ipython().system('pip install geopy')
import requests # library to handle requests
import pandas as pd # library for data analsysis
import numpy as np # library to handle data in a vectorized manner
import random # library for random number generation

#!conda install -c conda-forge geopy --yes 
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values

# libraries for displaying images
from IPython.display import Image 
from IPython.core.display import HTML 


from IPython.display import display_html
import pandas as pd
import numpy as np
    
# tranforming json file into a pandas dataframe library
from pandas.io.json import json_normalize

get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium # plotting library
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import matplotlib.colors as colors

print('Folium installed')
print('Libraries imported.')


# In[3]:


source = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M').text
soup=BeautifulSoup(source,'lxml')
print(soup.title)
from IPython.display import display_html
tab = str(soup.table)
display_html(tab,raw=True)


# In[39]:


df_1 = pd.read_html(tab)
df=df_1[0]
df.columns=["Postalcode","Borough","Neighbourhood"]
df.head()


# In[50]:


# Dropping the rows where Borough is 'Not assigned'
df=df[df['Borough']!='Not assigned']

df


# In[51]:


result = df.groupby(['Postalcode','Borough'], sort=False).agg( ', '.join)
df.loc[df['Neighbourhood'] =='Not assigned' , 'Neighbourhood'] = df['Borough']
df.head(10)


# In[52]:


df.shape


# In[ ]:




