import os
import xml.etree.ElementTree as ET

import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from centernet import CenterNet
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map

# dsaint31 ---------------------------------------------------
import os
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
os.environ['CUDA_VISIBLE_DEVICES']='-1'
# end --------------------------------------------------------

if __name__ == "__main__":
    '''
    Recall和Precision不像AP是一个面积的概念，因此在门限值（Confidence）不同时，网络的Recall和Precision值是不同的。
    默认情况下，本代码计算的Recall和Precision代表的是当门限值（Confidence）为0.5时，所对应的Recall和Precision值。

    受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算不同门限条件下的Recall和Precision值
    因此，本代码获得的map_out/detection-results/里面的txt的框的数量一般会比直接predict多一些，目的是列出所有可能的预测框，

    # celes --------------------------------------------------------
    Recall과 Precision은 AP와 달리 면적 개념이기 때문에 Confidence가 다를 때 네트워크의 Recall과 Precision 값은 다르다.
    기본적으로 이 코드에서 계산된 Recall과 Precision은 임계값(Confidence)이 0.5일 때 대응하는 Recall과 Precision값을 나타낸다.

    mAP 계산원리의 제약으로 인해 네트워크는 mAP 계산 시 거의 모든 예측 상자를 획득해야 서로 다른 임계값 조건에서의 Recall과 Precision 값을 계산할 수 있다.
    따라서, 본 코드에서 획득된 map_out/detection-results/안에 있는 txt는 일반적으로 직접 predict보다 더 많은 수의 박스를 가질 수 있으며, 모든 가능한 박스를 나열하는 것을 목적으로 한다.
    # end ----------------------------------------------------------

    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅获得真实框。
    #   map_mode为3代表仅仅计算VOC_map。
    #   map_mode为4代表利用COCO工具箱计算当前数据集的0.50:0.95map。需要获得预测结果、获得真实框后并安装pycocotools才行
    #-------------------------------------------------------------------------------------------------------------------#

    """
    map_mode 이 파일의 실행 시 계산을 지정하는 데 사용된다.
    map_mode는 0으로 전체 map 계산 흐름을 나타낸다. 포함예측 결과 획득, 리얼박스 획득, VOC_map 계산.
    map_mode는 1로 예측 결과만 얻었다.
    map_mode는 2로 실제 프레임만 획득한다.
    map_mode는 3으로 VOC_map만을 계산한다.
    map_mode 4는 COCO 툴박스를 이용하여 계산한다. 이전 데이터 세트의 0.50:0.95map.예측 결과를 얻고, 실제 박스를 얻은 후 pycocotools를 설치해야 함.

    """


    map_mode        = 0
    #--------------------------------------------------------------------------------------#
    #   此处的classes_path用于指定需要测量VOC_map的类别
    #   一般情况下与训练和预测所用的classes_path一致即可
    #--------------------------------------------------------------------------------------#

    """
    여기서 classes_path는 VOC_map을 측정할 범주를 지정한다.
    일반적으로 훈련과 예측에 사용되는 classes_path와 일치하면 된다.
    """

    classes_path    = 'model_data/ceph_voc_classes.txt' #dsaint31

    
    #--------------------------------------------------------------------------------------#
    #   MINOVERLAP用于指定想要获得的mAP0.x，mAP0.x的意义是什么请同学们百度一下。
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #
    #   当某一预测框与真实框重合度大于MINOVERLAP时，该预测框被认为是正样本，否则为负样本。
    #   因此MINOVERLAP的值越大，预测框要预测的越准确才能被认为是正样本，此时算出来的mAP值越低，
    #--------------------------------------------------------------------------------------#

    '''
    MINOVERLAP는 원하는 mAP0.x를 지정하는 데 사용된다.
    예를 들어 mAP0.75를 계산하면 MINOVERLAP= 0.75로 설정할 수 있다.

    예측 상자와 실제 상자의 일치도가 MINOVERLAP보다 클 경우, 예측 상자는 양의 표본으로 간주되고 그렇지 않을 경우 음의 표본으로 간주된다.
    따라서 MINOVERLAP의 값이 클수록 예측틀은 정확하게 예측해야 양의 샘플로 간주되며, 이때 산출되는 mAP 값은 낮을수록
    '''

    MINOVERLAP      = 0.5



    #--------------------------------------------------------------------------------------#
    #   受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算mAP
    #   因此，confidence的值应当设置的尽量小进而获得全部可能的预测框。
    #   
    #   该值一般不调整。因为计算mAP需要获得近乎所有的预测框，此处的confidence不能随便更改。
    #   想要获得不同门限值下的Recall和Precision值，请修改下方的score_threhold。
    #--------------------------------------------------------------------------------------#

    """
    mAP 계산원리의 제약으로 인해 네트워크는 mAP를 계산할 때 거의 모든 예측 상자를 획득해야 mAP를 계산할 수 있다.
    따라서 컨피던스의 값은 가능한 한 작게 설정해 가능한 모든 예측 박스를 얻어야 한다.

    이 값은 일반적으로 조정되지 않는다.mAP를 계산하려면 거의 모든 예측박스를 얻어야 하기 때문에 이곳의 컨피던스는 함부로 바꿀 수 없다.
    서로 다른 임계값에서 Recall과 Precision 값을 얻으려면 아래의 score_threhold를 수정할 것.
    """
    confidence      = 0.02



    #--------------------------------------------------------------------------------------#
    #   预测时使用到的非极大抑制值的大小，越大表示非极大抑制越不严格。
    #   
    #   该值一般不调整。
    #--------------------------------------------------------------------------------------#
    """
    예측에 사용된 비대칭 억제치의 크기는 커질수록 비대칭 억제가 엄격하지 않음을 나타낸다.
    이 값은 일반적으로 조정되지 않는다.
    """
    nms_iou         = 0.5




    #---------------------------------------------------------------------------------------------------------------#
    #   Recall和Precision不像AP是一个面积的概念，因此在门限值不同时，网络的Recall和Precision值是不同的。
    #   
    #   默认情况下，本代码计算的Recall和Precision代表的是当门限值为0.5（此处定义为score_threhold）时所对应的Recall和Precision值。
    #   因为计算mAP需要获得近乎所有的预测框，上面定义的confidence不能随便更改。
    #   这里专门定义一个score_threhold用于代表门限值，进而在计算mAP时找到门限值对应的Recall和Precision值。
    #---------------------------------------------------------------------------------------------------------------#

    """
    Recall과 Precision은 AP와 같은 면적 개념과는 달리 임계값이 다를 때 네트워크의 Recall과 Precision 값은 다르다.

    기본적으로 이 코드에서 계산된 Recall과 Precision은 임계값 0.5(여기서는 score_threhold로 정의됨)에 해당하는 Recall과 Precision 값을 나타낸다.
    mAP를 계산하려면 거의 모든 예측 상자를 얻어야 하기 때문에 위에서 정의한 컨피던스는 함부로 바꿀 수 없다.
    여기서 score_threhold를 정의하여 임계값에 대한 recall과 precision 값을 mAP 계산 시 찾는다.
    """
    score_threhold  = 0.5




    #-------------------------------------------------------#
    #   map_vis用于指定是否开启VOC_map计算的可视化

    #   Map_vis가 VOC_map 계산의 시각화를 켜는지 여부를 지정하는 데 사용된다.
    #-------------------------------------------------------#
    map_vis = True #dsaint31



    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集

    #   VOC 데이터 집합이 있는 폴더를 가리킨다.
    #   루트 디렉터리에 있는 VOC 데이터 집합을 기본값으로 지정됨
    #-------------------------------------------------------#
    VOCdevkit_path  = '../VOCdevkit' # dsaint31



    #-------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out

    #   결과 출력 폴더, 기본값은 map_out
    #-------------------------------------------------------#
    map_out_path    = 'map_out'



    image_ids = open(os.path.join(VOCdevkit_path, "ceph_VOC2007/ImageSets/Main/test.txt")).read().strip().split() # dsaint31 add ceph

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        centernet = CenterNet(confidence = confidence, nms_iou = nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "ceph_VOC2007/JPEGImages/"+image_id+".jpg") # dsaint31 add ceph
            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            centernet.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "ceph_VOC2007/Annotations/"+image_id+".xml")).getroot() # dsaint31 add ceph
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
