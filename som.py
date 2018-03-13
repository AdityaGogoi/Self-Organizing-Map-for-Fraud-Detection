# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
### Seperating last column of classification from the rest of the data.
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
# Converting all values between 0 and 1. Easier to compute.
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
### Importing MiniSom function from minisom.py file and training it on X
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)

### Initialising the weights and training the SOM
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()                          # Initialising the diagram screen 
pcolor(som.distance_map().T)    # Converting MID to color
colorbar()                      # Providing color legend
markers = ['o', 's']            # Defining marker shape
colors = ['r', 'g']             # Defining marker color
for i, x in enumerate(X):       # Looping over X, each row becoming a vector
    w = som.winner(x)           # Getting winning node for each row
    plot(w[0] + 0.5,            # Plotting marker at center of each node (x coordinate)
         w[1] + 0.5,            # and y coordinate
         markers[y[i]],         # Deciding marker shape based on output (marker[0] or marker[1])
         markeredgecolor = colors[y[i]],  # Getting  marker edge color based on output (color[0] or color[1])
         markerfacecolor = 'None',          # Putting no color for marker center          
         markersize = 10,                   # Defining size of each marker
         markeredgewidth = 2)               # Defining edge thickness of each marker
show()

# Finding the frauds
### Creating a dictionary of winning nodes and associated vectors
mappings = som.win_map(X)

### Gathering all the vectors under the 2 Winning nodes with high MID under the variable frauds.
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)











































