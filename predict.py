#----------------------------------------------------#
#   对视频中的predict.py进行了修改，
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#

"""
#----------------------------------------------------#
동영상 속의 predict.py을 수정했다.
단장 이미지 예측, 카메라 감지, FPS 테스트 기능
하나의 py파일에 통합돼 mode를 지정해 패턴을 수정한다.
#----------------------------------------------------#
"""

import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from centernet import CenterNet

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    """
    #-------------------------------------------------------------------------#
    mode는 테스트의 패턴을 지정하는 데 사용됩니다.
    mode : 
    - predict : 그림 저장, 대상 캡쳐 등 예측 과정을 수정하려면 아래쪽에 있는 상세한 주석을 먼저 보시기 바랍니다.
    - video : 비디오 검출을 나타내며, 카메라 또는 비디오를 호출하여 검출을 할 수 있으며, 자세한 내용은 아래의 주석을 참조.
    - fps : fps 테스트를 표시하며, 사용된 이미지는 img 안의 street.jpg이며, 자세한 내용은 아래의 주석을 참조.
    - dir_predict : 폴더를 탐색하여 저장. 기본적으로 img 폴더를 탐색하고 img_out 폴더를 저장함. 자세한 내용은 아래의 주석을 참조.
    - heatmap : 예측 결과의 열적 시각화를 나타내고, 자세한 내용은 아래의 주석을 참조.
    #-------------------------------------------------------------------------#
    """
    

    mode = "dir_predict"
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    """
    #-------------------------------------------------------------------------#
    - crop : Specify whether to capture the target after the leaflet image prediction
    - count : Specifies whether target count is performed
    - crop, count only mode : Valid for 'prediction'
    #-------------------------------------------------------------------------#
    """
    crop            = False
    count           = False




    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    """
    #-------------------------------------------------------------------------#
    - video_path : 비디오의 경로를 지정합니다. video_path=0일 때 카메라 감지
                    동영상을 검색하려면 video_path="xxx.mp4"와 같이 설정하면 되며 루트 디렉터리에 있는 xxx.mp4 파일을 읽습니다.
    - video_save_path : 비디오가 저장되는 경로를 나타냅니다. video_save_path="" 는 저장되지 않음을 나타냅니다.
                    동영상을 저장하려면 video_save_path="yyy.mp4"와 같이 설정하면 되며 루트 디렉터리에 있는 yyy.mp4 파일로 저장됩니다.
    - video_fps : 저장된 비디오의 fps
    #-------------------------------------------------------------------------#
    """
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0



    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    """
    #-------------------------------------------------------------------------#
    - test_interval : fps를 측정할 때 이미지 검출 횟수를 지정하는 데 사용한다. 이론상 test_interval이 클수록 fps는 정확하다.
    - fps_image_path : 테스트에 사용할 fps image 지정하기

    test_interval과 fps_image_path는 mode='fps'에서만 작동함!
    #-------------------------------------------------------------------------#
    """
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    """
    #-------------------------------------------------------------------------#
    - dir_origin_path : 검색할 그림의 폴더 경로를 지정
    - dir_save_path : 검색된 그림의 저장 경로를 지정

    dir_origin_path와 dir_save_path는 mode='dir_predict'일 때만 유효
    #-------------------------------------------------------------------------#
    """
    dir_origin_path = "../../Centernet/ceph/output/"
    dir_save_path   = "../../img_out/"



    #-------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #   
    #   heatmap_save_path仅在mode='heatmap'有效
    #-------------------------------------------------------------------------#\
    """
    #-------------------------------------------------------------------------#
    - heatmap_save_path : Save heatmap path, saved under model_data by default

    heatmap_save_path는 mode='heatmap'에서만 유효
    #-------------------------------------------------------------------------#
    """
    heatmap_save_path = "model_data/heatmap_vision.png"


    centernet = CenterNet(heatmap = True if mode == "heatmap" else False)
    
    if mode == "predict":
        '''
        1、如果想要进行检测完的图片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        2、如果想要获得预测框的坐标，可以进入centernet.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        3、如果想要利用预测框截取下目标，可以进入centernet.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        在原图上利用矩阵的方式进行截取。
        4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入centernet.detect_image函数，在绘图部分对predicted_class进行判断，
        比如判断if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        '''

        """
        1. If you want to save the detected image, you can save it using r_image.save ("img.jpg") and modify it directly at predict.py.
        2. To get the coordinates of the prediction box, go to the cenet.detect_image function and read the top, left, Bottom, and right values in the drawing section.
        3. If you want to capture a target using the prediction box, go to the center.detect_image function and capture the four values of top, left, Bottom,
            and right on the drawing section using the matrix.
        4. If you want to write additional words on a prediction chart, such as the number of specific targets detected, go to the cenet.detect_image function
            and judge the predicted_class in the drawing section.
            For example, determine if predicated_class=='car': to determine whether the current target is a vehicle,
            and then record the number.Draw.text can be used to write.
        """
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = centernet.detect_image(image, crop = crop, count=count)
                r_image.show()


    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            #raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
            raise ValueError("The camera (video) was not read correctly. Be careful that the camera is installed correctly (video path is entered correctly).")

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            # Read frame
            ref, frame = capture.read()
            if not ref:
                break

            # 格式转变，BGRtoRGB
            # Formatting, BGR to RGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # 转变成Image
            # Image from array
            frame = Image.fromarray(np.uint8(frame))

            # 进行检测
            # Test
            frame = np.array(centernet.detect_image(frame))

            # RGBtoBGR满足opencv显示格式
            # RGB to BGR
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
        
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = centernet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = centernet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                centernet.detect_heatmap(image, heatmap_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
