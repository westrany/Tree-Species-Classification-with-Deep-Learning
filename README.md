# Tree Image Classification with Deep Learning

_Note: the following project is an academic assignment part of my Data Science Bachelor's at the University of Stirling._

### Introduction:

Classification is a machine learning model used to predict mutually exclusive categories when there are no continuous values. It does so by using labels to classify data, thus predicting a discrete number of values.  

Deep Learning is one of many of machine learning methods which bases itself on artificial neural networks with multiple layers of processing that are used to extract progressively higher level features from data; such can be supervised, semi-supervised or unsupervised.  

This Contextual Image Classification Project makes use of Deep Learning to train a Classification model to recognize and correctly classify trees in an urban context. Deep Learning models often require a considerable amount of relevant training data to be able to provide good results, thus the collection of 400 eye-picked images that depict trees in different settings, seasons and lightings. When compiling this dataset, I had extra care to ensure the images I was collecting were either public domain or free to use as long as the owner is ackowledged; to ensure this, I created a csv file with the sources for each image. 

***  
### Problem Definition  

This image classification project aims to recognize and classify trees in different urban settings, as well as correctly classify them according to species.  

A research paper [[1](https://link.springer.com/article/10.1007/s11676-020-01245-0)] was used as reference on how to train the classification model on species.  

***

### Dataset:

Trees vary in shape, colour, density and folliage, which can become a struggle for AI to determine a pattern as the model will have to be trained to recognize intra-class variations (different species of trees), scale variations (trees have different sizes), perspective variation (images portray trees from different angles), different illumination levels (influenced by different stages of the day such as morning, dusk, or night), and background cluttler, to name a few.  

To better train my module on this, I chose to collect image data from two different urban environments with different biomes and cultural settings, thus having two cities (A and B, with 200 images each) which will provide enough diversity in terms of tree species, scale, perspective, illumination and clutter, strengthening my model to more accurate and precise levels.  

• City A is my birthtown, Ponta Delgada (São Miguel, Azores, PT), with an Oceanic climate and Azorean Macaronesian Flora (also refered to as the Laurissilva Forest) with over 60 unique plants (special emphasis on Cedar (_Arceuthobium azoricum_), "Pau Branco" (_Picconia azorica_) and Laurel tree (_Laurus azorica_)).  

• City B is my current city of residence, Stirling (Scotland, UK) as well with an Oceanic climate but with a different Flora constituted by heather moorland, coastal machair and a reduced boreal Caledonian forest. The similarity in climate (both cities have Oceanic climate) can benefit the AI model as tree species will have some similarities in vegetation; their different Floras, however, can complicate the classification problem as the AI will have to learn to identify more varieties of trees, which in itself is a perk as it will train the module to recognize a broader scope of tree species.  

I have chosen to work with a batch size of 64 images. This is a rather small number which will result in noisy updates to model weights; that is to say that there will be many updates with different estimates of the gradient error. This can be useful, resulting in faster learning and (sometimes) a more robust model as noisier batch sizes offer better regularizing effects, lower generalization error, and make it easier to fit one batch worth of training data in GPU memory.

***

### Implementation: 

The first step involves object detection, a challenging problem that requires methods for "(e.g. where are they), object localization (e.g. what are their extent), and object classification (e.g. what are they)" [4]. To do so, I have chosen to use R-CCN.  

Region-Based Convolution Neural Network (R-CNN)

***

### Conclusion:
[to be added]

***

### File Index:  

- [Dataset](https://github.com/westrany/CSCU9M6-2929300/tree/main/Dataset): folder with image dataset 
  - [CityA_images](https://github.com/westrany/CSCU9M6_Tree-Classification-in-Ponta-Delgada-vs-Stirling/tree/main/CityA_images): 200 image database from City A
  - [CityB_images](https://github.com/westrany/CSCU9M6-2929300/tree/main/CityB_images): 200 image database from City B
- [ImageSource.csv](https://github.com/westrany/CSCU9M6_Tree-Classification-in-Ponta-Delgada-vs-Stirling/blob/main/ImageSource.csv): a database with image sources  
- [2023_Spring_Assignment_M6.ipynb](https://github.com/westrany/CSCU9M6_Tree-Classification-in-Ponta-Delgada-vs-Stirling/blob/main/2023_Spring_Assignment_M6.ipynb): main Jupyter Notebook with project instructions and assembled code

***  

### Research Sources:  

[1] C. Zhang, K. Xia, H. Feng, Y. Yang, and X. Du, [“Tree species classification using deep learning and RGB optical images obtained by an unmanned aerial vehicle,”](https://link.springer.com/article/10.1007/s11676-020-01245-0) Journal of Forestry Research, vol. 32, no. 5, pp. 1879–1888, 2020. 

[2] J. Brownlee, “How to control the stability of training neural networks with the batch size,” MachineLearningMastery.com, 27-Aug-2020. [Online]. Available: https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/. [Accessed: 24-Mar-2023]. 

[3] K. Team, “Keras documentation: Introduction to keras for engineers,” Keras. [Online]. Available: https://keras.io/getting_started/intro_to_keras_for_engineers/. [Accessed: 25-Mar-2023].  

[4] J. Brownlee, “How to train an object detection model with keras,” MachineLearningMastery.com, 01-Sep-2020. [Online]. Available: https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/. [Accessed: 25-Mar-2023]. 

[5] Matterport, “Mask R-CNN for Object Detection and Segmentation,” GitHub, 19-Mar-2018. [Online]. Available: https://github.com/matterport/Mask_RCNN. [Accessed: 25-Mar-2023].   

[6] “Image classifier using CNN,” GeeksforGeeks, 11-Jan-2023. [Online]. Available: https://www.geeksforgeeks.org/image-classifier-using-cnn/. [Accessed: 25-Mar-2023]. 

***

### License:  

This project follows MIT Lience - see [LICENSE](https://github.com/westrany/CSCU9M6_Tree-Classification-in-Ponta-Delgada-vs-Stirling/blob/main/LICENSE) for more details. Due to the library wraps used, other free license types might be inherited.
