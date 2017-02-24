import numpy as np
import pandas as pd

def create_train_with_target(df, products):
    for step in range(1,6):
        columns = list(map(lambda x: x + '_prev' + str(step), products))
        df[columns] = df[columns].fillna(0).astype(np.int8)

    data = []
    for i, product in enumerate(products):
        prev = product + "_prev1"
        df.loc[(df[product] == 1) & (df[prev] == 0), 'target'] = i
        for idx, row in df[(df[product] == 1) & (df[prev] == 0)].iterrows():
            data.append(row)

    df = pd.DataFrame(data, columns=list(df.columns.values))
    return df[df['target'].notnull()]
