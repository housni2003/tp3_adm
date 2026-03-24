import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib

# Load the data
with open('villes.csv', 'r') as f:
    content = f.read()
if ';' in content and '\n' not in content and '\r' not in content:
    # If it's literally one line and not separated properly (which seems to be the case from view_file output but let's see)
    pass

df = pd.read_csv('villes.csv', sep=';', index_col=0)
print(df.head())
