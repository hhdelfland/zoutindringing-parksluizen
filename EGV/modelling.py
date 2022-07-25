import os
import pandas as pd

def mdl_get_feats(index):
    files = []
    for file in os.listdir('data_sets'):
        if file.startswith('feats_') and file.endswith('.csv'):
            files.append(file)
    dataset = pd.read_csv('data_sets/'+files[index])
    dataset = dataset.set_index('datetime', drop=False)
    return dataset


def main():
    print(mdl_get_feats(1))


if __name__ == '__main__':
    main()
