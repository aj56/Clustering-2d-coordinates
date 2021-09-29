import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

def clustering(df):
    kmeans = KMeans(2)
    kmeans.fit(df)
    print("Centroid Values are :",kmeans.cluster_centers_)
    identified_clusters = kmeans.fit_predict(df)
    print("Cluster ID's for the respective points are",identified_clusters)
    data_with_clusters = df.copy()
    data_with_clusters['Clusters'] = identified_clusters
    plt.scatter(data_with_clusters['x'],data_with_clusters['y'],c=data_with_clusters['Clusters'],cmap='rainbow')
    plt.show()

def getinput():
    inp = [(1,3),(7,2),(4,6),(34,3),(33,1)]  #input value based on question
    lis1 = []
    lis2 = []
    for i in range(len(inp)):
        lis1.append(inp[i][0])
        lis2.append(inp[i][1])
    data = [lis1,lis2]
    df = pd.DataFrame (data).transpose()
    df.columns = ['x','y']
    plt.scatter(df['x'],df['y'])
    plt.show()
    return df

if __name__ == "__main__":
    df = getinput()   
    clustering(df)
