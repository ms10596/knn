import pandas as pd
import numpy as np


def read_data(file_name):
    data = pd.read_csv(file_name)
    x = np.array(
        [data['x0'], data['x1'], data['x2'], data['x3'], data['x4'], data['x5'],
         data['x6'], data['x7']]).T
    y = np.array(data['y'])
    return x, y