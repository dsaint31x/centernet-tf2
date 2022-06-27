# 이미지 경로에서 이미지들을 불러와 txt로 자동 저장하는 기능


# label_data : point 정보가 담긴 csv file
def textwrite(label_data):
    image_array = []

    print('Directory has been created.')

    fp_all = open('./ImageSets/Layout/all.txt','wt')
    fp_train = open('./ImageSets/Layout/train.txt','wt')
    fp_val = open('./ImageSets/Layout/val.txt','wt')
    fp_test = open('./ImageSets/Layout/test.txt','wt')


    for idx,c in enumerate(label_data['ID']):
        if idx < 550:
            fp_train.write(f'{c}\n')
            fp_all.write(f'{c}\n')
        elif idx < 600:
            fp_val.write(f'{c}\n')
            fp_all.write(f'{c}\n')
        elif idx >= 600:
            fp_test.write(f'{c}\n')
            fp_all.write(f'{c}\n')
    
        
    fp_all.close()
    fp_train.close()    
    fp_val.close()
    fp_test.close()