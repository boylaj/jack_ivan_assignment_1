#!/usr/bin/env python
# coding: utf-8

# # Exploring & Visualising Data

# ## Reading in our data and Importing libraries

# In[1]:


get_ipython().system('pip3 install plotly')
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as pyo
pyo.init_notebook_mode()
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
get_ipython().run_line_magic('matplotlib', 'inline')
from warnings import filterwarnings
filterwarnings('ignore')

df1 = pd.read_csv("data/parameters_igt_vpp.csv",sep=",")

df1.head()


# ## Features
# Our features in this dataset are the parameters of the vpp model. These are:
# 
# * Learning Rate
# * Outcome Sensitivity
# * Response Consistency
# * Loss Aversion
# * Gain Impact
# * Loss Impact
# * Learning Decay Rate
# * Reward Learning Rate

# In[2]:


labels = df1.group
cols = df1.columns[2:]
cols


# ## Standardising Our Data
# 
# In order for us to use Principal Component Analysis on our dataset, we first must standardise our data so that each feature is weighted equally.

# In[3]:


standardised = scipy.stats.zscore(df1[cols])
df1_std = pd.DataFrame(standardised, columns = cols)
df1_std


# ## Principal Component Analysis
# 
# PCA is a method of feature reduction. When dealing with large datasets that have many features, clustering algorithms often struggle to be effective due to the [Curse of Dimensionality.](https://en.wikipedia.org/wiki/Curse_of_dimensionality)
# 
# It is also difficult to visualise data in high dimensions, and so we may wish to condense our dataset into a mangeable number of features, while still retaining as much information about the data as possible.
# 
# PCA allows us to achieve this by finding Principal Components - axes which capture as much variance from the data as possible.
# 
# 
# ## Finding Number of Principal Components
# 
# The number of Principal Components n we want the data to be reduced to will need to be given by us. In order to find a suitable n, we must find the amount of variance captured in each PC and choose n that provides >=75% variance.
# 
# We make use of the following function to plot the amount of variance captured at each value of n:

# In[4]:


# use this function to find variance captured in n PCs
# ideally we'd like 75-90% variance captured in 2 or 3 PCs
# to give an accurate representation of the original data

def ideal_pca(df_std):
    dims = len(df_std.columns)
    pca = PCA(n_components = dims)
    pca.fit(df_std)
    variance = pca.explained_variance_ratio_
    variance = np.insert(variance, 0, 0, axis=0)
    var = np.cumsum(np.round(variance, 3)*100)
    plt.figure(figsize=(12,6))
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Principal Components')
    plt.title('PCA Analysis')
    plt.ylim(0,100.5)
    plt.xlim(0, dims)
    plt.plot(var)
    
ideal_pca(df1_std)


# **We can see that 3 Principal Components are within our desired range.**
# ### We will fit our data to 3 PCs

# In[5]:


pca3 = PCA(n_components = 3).fit(df1_std)
pca3d = pca3.transform(df1_std)
pca3d_df = pd.DataFrame(pca3d, columns=["PC1", "PC2", "PC3"])

pca3d_df["group"] = df1.group

print(pca3.explained_variance_ratio_)

# the % variance captured by our PCs


# In the following 3D plot, we can visualise our entire dataset through our Principal Components. We can clearly see the young and old participants in two distinct groups after PCA.

# In[6]:


# plot all 3 PCs

Scene = dict(xaxis = dict(title  = 'PC1'),
             yaxis = dict(title  = 'PC2'),
             zaxis = dict(title  = 'PC3'))

trace = go.Scatter3d(x=pca3d[:,0],
                     y=pca3d[:,1],
                     z=pca3d[:,2],
                     mode='markers',marker=dict(color = pca3d_df["group"].astype("category").cat.codes,
                                                colorscale='Viridis', size = 10,
                                                line = dict(color = 'gray',width = 5)))
layout = go.Layout(margin=dict(l=0,r=0),
                   scene = Scene,
                   height = 1000, width = 1000)
data = [trace]

# Uncomment the following line to display interactive graph when not in Jupyter Book
#pyo.iplot(data, filename = 'basic-scatter1.html')


# ![plot](newplot.png)

# ## Exploration using t-SNE
# 
# T-SNE is another dimensionality reduction technique, although it differs from PCA in a number of ways. 
# 
# T-SNE has the ability to reduce dimensions using similarity measures based on distances between points. It is often a helpful visualisation tool as it can reduce dimensions with non-linear relationships, an advantage over PCA.
# 
# However, there are a number of parameters we must pass to fit t-SNE to our data. 
# 
# This next function helps find ideal perplexity for the data [(somewhere between 5 - 55)](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf). Perplexity specifies the number of points t-SNE will consider as a cluster, and the optimal value depends on the data.
# 
# We must run this function a few times, as t-SNE is a probabalistic model, which means it starts from a random state, and we want a stable shape to emerge from our iterations.

# In[7]:


def find_ppl(df_std, dims):
    ppl = 5
    while ppl < 60:
        tsne = TSNE(n_components=2, verbose=1, perplexity=ppl, n_iter=5000, learning_rate=200)
        tsne_scale_results = tsne.fit_transform(df1_std)
        tsne_df_scale = pd.DataFrame(tsne_scale_results, columns=['tsne1', 'tsne2'])
        plt.figure(figsize = (10,10))
        plt.scatter(tsne_df_scale.iloc[:,0],tsne_df_scale.iloc[:,1],alpha=0.8, facecolor='lightslategray')
        plt.xlabel('tsne1')
        plt.ylabel('tsne2')
        plt.show()
        print("Perplexity of {}".format(ppl))
        ppl += 10
        
find_ppl(df1_std, len(cols))
# looking at these plots, ppl of 25 seems to give the same shape every time


# In[8]:


tsne = TSNE(n_components=2,
            verbose=1,
            perplexity=25,
            n_iter=5000,
            learning_rate=200)
tsne_scale_results = tsne.fit_transform(df1_std)
tsne_df_scale = pd.DataFrame(tsne_scale_results, columns=['tsne1', 'tsne2'])


# In[9]:


plt.figure(figsize = (15,15))
sns.scatterplot(tsne_df_scale.iloc[:,0],
                tsne_df_scale.iloc[:,1],hue=df1.group,
                palette='Set1',
                s=100,
                alpha=0.6).set_title('tSNE Visualising Standardised Data', fontsize=15)
plt.legend()
plt.show()


# ## Characteristics of each group
# In the following histograms, we can see the distribution of feature values among our old and young participants using both PCA and t-SNE.

# In[10]:


for c in df1:
    grid = sns.FacetGrid(df1, col='group')
    grid.map(plt.hist, c)


# Our visualisations show that young people are much more sensitive to the outcome of their chosen deck.
# Young people show more varied response consistency, being more experimental while choosing decks.
# 
# Older players are more heavily impacted by gains, and less affected by losses relative to the younger participants.
# 
# Younger participants present a much higher reward learning rate compared to older subjects.

# In[11]:


plt.figure(figsize=(15,10))
sns.scatterplot(data=df1, x='Loss_Impa', y='Gain_Impa', 
                hue='group', s=85, alpha=0.7, palette='bright').set_title(
    'Age Group by Loss Impact and Gain Impact',fontsize=18)


# We can see that there is a weak pos correlation between gain and loss impact.
# 
# People who react more to wins, also react more to losses, regardless of age.
# 
# In the 3D plot below, we can further visualise the decision making qualities that separate the old and young participants.

# In[12]:


# 3d plot clusters comparing response consitency, outcome sensitivity and loss aversion
Scene = dict(xaxis = dict(title  = 'Response Consistency'),
             yaxis = dict(title  = 'Outcome Sensitivity'),
             zaxis = dict(title  = 'Loss Aversion'))
labels = df1["group"].astype("category").cat.codes
trace = go.Scatter3d(x = df1["Res_Cons"],
                     y = df1["Out_Sens"],
                     z = df1["Loss_Aver"],
                     mode='markers',
                     marker=dict(color = labels, colorscale='Viridis', size = 10, line = dict(color = 'gray',width = 1)))

layout = go.Layout(margin=dict(l=0,r=0),
                   scene = Scene,
                   height = 800,width = 800)
data = [trace]

# Uncomment the following line to display interactive graph when not in Jupyter Book
#pyo.iplot(data, filename = 'basic-line2')


# ![plot](newplot2.png)
