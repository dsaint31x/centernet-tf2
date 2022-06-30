import pandas as pd
import numpy as np


### csv load
inf_csv = pd.read_csv('./csv/out.csv')
gt_csv = pd.read_csv('./csv/gt.csv')


# inferance에 사용된 landmark 수
landmarks_num = 46


for i in range(650):
    predict_point = []
    for j in range(landmarks_num):
        x, y = inf_csv([i][j])
        gt, gy = gt_csv[i]