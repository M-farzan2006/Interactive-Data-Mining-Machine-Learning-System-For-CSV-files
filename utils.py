import pandas as pd
import matplotlib.pyplot as plt

def load_csv(path):
    return pd.read_csv(path)

def save_plot(df, clusters):
    plt.figure(figsize=(5, 4))
    plt.scatter(df.iloc[:,0], df.iloc[:,1], c=clusters)
    plt.title("Clusters Visualization")
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.savefig("outputs/clusters_plot.png")
    plt.close()
