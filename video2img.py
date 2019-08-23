import cv2
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

video_path = '/home/shixi/C3D-keras/datasets/ucf101/'
save_path = '/home/shixi/C3D-keras/datasets/ucfimgs/'

action_list = os.listdir(video_path)

for action in action_list:
    if not os.path.exists(save_path+action):
        os.mkdir(save_path+action)
    video_list = os.listdir(video_path+action)
    for video in video_list:
        prefix = video.split('.')[0]
        if not os.path.exists(save_path+action+'/'+prefix):
            os.mkdir(save_path+action+'/'+prefix)
        save_name = save_path + action + '/' + prefix + '/'
        #save_name = save_path + prefix + '/'
        video_name = video_path+action+'/'+video
        name = video_name.split('.')[1]
        if name == "avi":
            cap = cv2.VideoCapture(video_name)
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_count = 0
            for i in range(fps):
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(save_name + str(10000 + fps_count) + '.jpg', frame)
                    fps_count += 1
        if name == "gif":
           im = Image.open(video_name)
           #当打开一个序列文件时，PIL库自动加载第一帧。你可以使用seek()函数tell()函数在不同帧之间移动。实现保存
           try:
              while True:
                   current = im.tell()
                   img = im.convert('RGB')  #为了保存为jpg格式，需要转化。否则只能保存png
                   img.save(save_name+'/'+str(10000+current)+'.jpg')
                   im.seek(current + 1)
           except EOFError:
               pass
