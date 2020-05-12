import numpy as np
import csv

def get_data():
    with open("../data/combined.csv") as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = np.array([row for row in reader])
    countys = data[:, 2]    
    data = np.delete(data, 2, axis = 1)
    data = data.astype('float64')
    return headers, data[:,:-1], data[:,-1][:,np.newaxis]
