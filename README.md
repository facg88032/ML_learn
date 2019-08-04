ML_learn
===
ML_learn is some simply manual model , Not use any tools .

Object detection
===
Object detection use [Tensorflow Objection Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) to try various topic(face,poker,star....) .

 I learn from  [TensorFlow-Object-Detection-API-Tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) ,  this tutorial teaches how to label image ,train and application.

## Label Image
LabelImg can label that you expect to detection objects in every picture.<br> its GitHub page has very clear instructions on how to install and use it.<br>
[LabelImg link](https://github.com/tzutalin/labelImg)

#### Ex:
![image](https://github.com/facg88032/picture/blob/master/labelImage.png)

## Train
TensorFlow provides several object detection models (pre-trained classifiers with specific neural network architectures) in its [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). you can try different models that have some different properties.<br>
I use Faster_RCNN_Resnet101_coco ,because this model trains faster and accuracy performance is great.
## Application
You can use the webcam property object detection , you also can detecton object on image or video .
#### Ex:

![image](https://github.com/facg88032/picture/blob/master/webcam_person2.png)<br>
Single object detection with webcam <br>

![image](https://github.com/facg88032/picture/blob/master/multi_person_webcam.jpg)<br>
Multiple Objects detection with webcam <br>

![image](https://github.com/facg88032/picture/blob/master/image_person1.png)<br>
Object detection on Image <br>
