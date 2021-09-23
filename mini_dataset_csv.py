import os
import pandas as pd


def main():
    data_root = 'processed_data'
    phases = [('train', ['M011', 'W014']), ('val', ['M009', 'W011']), ('test', ['M003', 'W009'])]

    for phase, selected in phases:
        csv_file = os.path.join(data_root, f'{phase}_subjects.csv')
        mini_csv = os.path.join(data_root, f'{phase}_mini.csv')
        train_subjects = pd.read_csv(csv_file)
        firsts = train_subjects[train_subjects['audio'] == '001.wav']
        firsts = firsts[firsts['subject'].isin(selected)]
        firsts.to_csv(mini_csv, index=False)


if __name__ == '__main__':
    main()