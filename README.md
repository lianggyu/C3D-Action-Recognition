# C3D-Action-Recognition
Train the C3D network with my own data set. Video or gif can be supported as a training file. Video streams or image frames can be used as input for detection.

## Environment
* opencv-3.2
* keras-2.0.8
* tensorflow-1.3
## Train your own data
* Place the data in the `datasets/ucf101`. The label for training (train_file.txt), label for testing (test_file.txt), are placed in the `/ucfTrainTestlist`. Record category in classInd.txt.

* Run `video2img.py` and save it in the datasets/ucfimgs. Currently gif and video formats are supported.

* Run `make_label_txt.py` and generate train_list.txt and test_list.txt.

* Modify `model.py`, such as lines 7~9.

* Modify train_c3d.py, such as lines 158~164. Run `train_c3d.py`.
## Demo
* Modify `config.txt`<br>
  classInd_path : the file path of the record category<br>
  lr : learning rate (default 0.005)<br>
  momentum : Momentum (default 0.9)<br>
  weights_path : path to the trained model<br>
  
  Image mode:<br>
  image_read : the path to the image read<br>
  image_write : the path written by the image<br>
  
  video_image: select input mode<br>
  video: the mode of video input<br>
  image: the mode of image input<br>
  <br>
* Modify `video_demo.py`<br>
  If you want to apply it to the video stream, you must first modify the dictionary(video_stream).<br>
  video_stream[key].[0] : This is an array that records 16 video frames as input to the model.<br>
  video_stream[key].[1] : the path to the data.<br>
  video_stream[key].[2] : the path of the model.<br>
  <br>
 * Run `video_demo.py`<br>
 Identification of abnormal-event and normal-event:<br>
 <img src="https://github.com/lianggyu/C3D-Action-Recognition/blob/master/results/frame_1.png" width="300" alt="normal-event"/>     <img src="https://github.com/lianggyu/C3D-Action-Recognition/blob/master/results/frame_2.png" width="387" alt="abnormal-event"/><br>
 Identification of smoking:<br>
 <img src="https://github.com/lianggyu/C3D-Action-Recognition/blob/master/results/frame_3.png" width="300" alt="normal-event"/>     <img src="https://github.com/lianggyu/C3D-Action-Recognition/blob/master/results/frame_4.png" width="290" alt="abnormal-event"/><br>
