# Iowa Gambling Task - Model Parameter Analysis

## Introduction

In this assignment we showcase our methods used to analyse and cluster results from 3 separate reinforcement learning models used to simulate human behaviour in relation to the Iowa Gambling Task.

**The Iowa Gambling Task** is a psychological task thought to simulate real-life decision making. At each turn, the participants choose one of four decks of cards.

Each deck will either punish or reward the player with smaller and larger amounts of money. Some decks will give the player a net loss on average, while other decks will provide a net gain. How the players interact with the decks is recorded and used to analyse behavioural characteristics.

The three reinforcement learning models used to simulate human behaviour were:

1. Outcome Representation Learning (ORL) model
2. Values-Plus-Perseverance (VPP) model
3. Prospect Valence Learning model with Delta (PVL-Delta)

## Our Aim

We have been provided with three datasets containing parameter values from these models and have been tasked with identifying clusters from the data.

We plan to use PCA and t-SNE to explore visualisations of our data, and utilise a number of clustering algorithms to identify the number of clusters that best represent our data and how these clusters are chosen.

 We also plan to use Spectral Clustering to build relationships between simulated individuals in order to discover possible hidden clusters using KMeans.
