import pandas as pd


def PSWQ_Dutch_Positve():
    data = pd.read_csv('PSWQ_Dutch.csv', sep=',')
    # Will hold question configurations from file.
    PSWQ = data.filter(like='PSWQ_', axis=1)
    # Remove reverse coded questions.
    PSWQ.drop(PSWQ.columns[[0, 2, 7, 9, 10]], axis=1, inplace=True)

    return PSWQ