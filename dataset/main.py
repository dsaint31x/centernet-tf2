from txt_generate import textwrite
from xml_generate import xml_writer

import pandas as pd, os

def __init__(self):
    self.label_data = None

def main():
    # csv load
    label_data = pd.read_csv('./label.csv')
    print('[Log] csv file has been loaded.')

    os.makedirs('./Annotations/', exist_ok=True)
    os.makedirs('./ImageSets/Layout/', exist_ok=True)

    textwrite(label_data)
    xml_writer(label_data)

    print('[Log] Done')


main()