# video-classification
A project re-evaluating 2D CNNs for activity recognition.

I thank PyImageSearch for this interesting tutorial on Video Classification using 2D CNNs which has been a major inspiration for this work. The tutorial can be found here:
https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/

The custom dataset: This work makes use of 2D CNNs and hypothesises that 2D CNNs can be trained on images representing different facets of human activities.
The reason 2D CNNs are not used for video classification tasks is that they are not capable of learning or extracting spatio-temporal features owing to the
absence of 3D computational kernels. This work however counters this fact using rolling prediction averaging. The average of confidence scores of a series of frames
is taken and the output label having the highest confidence score is displayed as output. The training of the models is therefore done on images that respresent different
facets of human activities. 

The models that were tested are VGG-16, ResNet 50 and MobileNetv2, all of which are ImageNet pretrained models. 

The models have ImageNet weights. Transfer Learning was used to train these models on the custom dataset. Rolling prediction averaging is done while testing the models on
real-time videos. The number of frames that need to be averaged upon is upto the user. Higher the number of frames taken into the buffer for prediction averaging, closer is
the model's performance to deep 3D CNNs trained on video datasets for activity recognition. There are certainly areas where 3D CNNs are just the right choice for video classification.

The custom dataset has not been added here. However, usesrs can download different images pertaining to an activity like cricket, online via a simple Google search.
Images of different activities need to be put inside separate folders named after the activity. The number of neurons in the softmax layer will have to updated to be the same 
as the number of activities that are part of your dataset.


