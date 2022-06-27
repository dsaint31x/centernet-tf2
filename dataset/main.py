from txt import textwrite
from xml import xml_writer

import pandas as pd

def __init__(self):
    self.label_data = None

def main():
    # csv load
    label_data = pd.read_csv('csv path')
    
    if label_data==None:
        raise ValueError('You must set an appropriate path.')

    textwrite(label_data)
    xml_writer(label_data)