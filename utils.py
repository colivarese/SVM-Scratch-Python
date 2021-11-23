import pandas as pd
import numpy as np 

def loadData(path:str, outcome_attr:str):
    df = pd.read_csv(path)


    outcome_values = df[outcome_attr].unique()
    map_dict = dict()
    values = [-1,1]
    for i in range(len(outcome_values)):
        print(outcome_values[i] ,' was converted to ', values[i])
        map_dict[outcome_values[i]] = values[i]
    df[outcome_attr] = df[outcome_attr].map(map_dict)

    outcome = df[outcome_attr]
    df.drop(outcome_attr, axis=1, inplace=True)

    n = len(df)
    train_n = int(np.round(n*0.70))

    train_data, train_labels = df.loc[train_n:], outcome.loc[train_n:]
    test_data, test_labels = df.loc[:train_n], outcome.loc[:train_n]
    
    return train_data, train_labels, test_data, test_labels
