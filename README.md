# Tree Species Classification with Deep Learning

![image](https://github.com/westrany/Tree-Species-Classification-with-Deep-Learning/assets/69496007/a50015c2-28f0-41f0-8dc7-ee33385aff02)

_Note: the following project was adapted from an academic assignment part of my Data Science Bachelor's at the University of Stirling._

### Introduction:

Classification is a machine learning model used to predict mutually exclusive categories when there are no continuous values. It does so by using labels to classify data, thus predicting a discrete number of values.  

Deep Learning is one of many of machine learning methods which bases itself on artificial neural networks with multiple layers of processing that are used to extract progressively higher level features from data; such can be supervised, semi-supervised or unsupervised.  

This Contextual Image Classification Project makes use of Deep Learning to train a Classification model to recognize and correctly classify trees in an urban context. Deep Learning models often require a considerable amount of relevant training data to be able to provide good results, thus the collection of 520 eye-picked images that depict trees in different settings, seasons and lightings. When compiling this dataset, I had extra care to ensure the images I was collecting were either public domain or free to use as long as the owner is ackowledged; to ensure this, I created a csv file with the sources for each image. 

***  
### Problem Definition  

This image classification project aims to recognize and classify trees species in different urban settings.

A research paper [1] was used as reference on how to train the classification model on species.  

***

### Dataset:

Trees vary in shape, colour, density and folliage, which can become a struggle for AI to determine a pattern as the model will have to be trained to recognize intra-class variations (different species of trees), scale variations (trees have different sizes), perspective variation (images portray trees from different angles), different illumination levels (influenced by different stages of the day such as morning, dusk, or night), and background cluttler, to name a few.  

To better train my module on this, I chose to collect image data from two different urban environments with different biomes and cultural settings, thus having two cities which will provide enough diversity in terms of tree species, scale, perspective, illumination and clutter, strengthening my model to more accurate and precise levels.  

• City A is Ponta Delgada (São Miguel, Azores, PT), with an Oceanic climate and Azorean Macaronesian Flora (also refered to as the Laurissilva Forest) with over 60 unique plants (special emphasis on Cedar, Plane Tree and Araucaria).

• City B is Stirling (Scotland, UK) as well with an Oceanic climate but with a different Flora constituted by heather moorland, coastal machair and a reduced boreal Caledonian forest. The similarity in climate (both cities have Oceanic climate) can benefit the AI model as tree species will have some similarities in vegetation; their different Floras, however, can complicate the classification problem as the AI will have to learn to identify more varieties of trees, which in itself is a perk as it will train the module to recognize a broader scope of tree species.

The training data set is composed of 120 images, 60 from each city, and has 15 labels, 14 containing trees species (7 labels with 8 images for each city) and 1 corresponding to images without trees (4 images from each city).  

*Both training and testing data sets have a rather small number which will result in noisy updates to model weights; that is to say that there will be many updates with different estimates of the gradient error. This can be useful, resulting in faster learning and (sometimes) a more robust model as noisier batch sizes offer better regularizing effects, lower generalization error, and make it easier to fit one batch worth of training data in GPU memory.*

***  

### Labels:  

|**#**   |**City**   |**Species**   |**Label**   |
|:-:|:-:|:--:|:--:|
|1   |A & B|no tree in image   | not  | 
|2  |A   |araucaria               |ara   |
|3 |A   |banana tree                |ban   |
|4|A   |chestnut                 |che   |
|5   |A   |palm tree        |pal   |
|6   |A   |plane tree           |pla   |
|7   | A  |australian rubber tre   |art  |
|8   |A   |cedar                   |ced   |
|9   | B  |ash tree           |ash   |
|10   | B  |sycamore    |     syc  |  
|11   | B |oak   |oak   |
|12   | B  |apple tree   |app   |
|13   | B  |Scots pine   |pin  |
|14   | B  |beech   |bee   |  
|15   | B |bird cherry tree   |bct   |


***

### Libraries Used:  

**Keras:** a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. Keras is designed to enable fast experimentation with deep neural networks, making it easy to build and train models.

**Matplotlib:** a comprehensive library for creating static, animated, and interactive visualizations in Python. It is particularly useful for generating plots, histograms, bar charts, and other types of data visualizations.

**Numpy:** a fundamental package for scientific computing in Python. It provides support for arrays, matrices, and a wide range of mathematical functions, making it essential for numerical computations. 

**os:** a standard library in Python that provides a way to use operating system-dependent functionality like reading or writing to the file system. It allows for the manipulation of files and directories, environment variables, and other system-level operations.

**PLI (Python Imaging Library):** an external library (now succeeded by Pillow) for opening, manipulating, and saving many different image file formats. It is used for tasks such as image processing, including resizing, cropping, and filtering.

**scikit-learn:** an open-source machine learning library that features various classification, regression, and clustering algorithms. It is built on NumPy, SciPy, and Matplotlib, providing simple and efficient tools for predictive data analysis.

**TensorFlow:** an open-source platform for machine learning developed by Google. It provides a comprehensive ecosystem for building and deploying machine learning models, particularly deep learning models, across various platforms and devices.

***

### Dataloader: 

The first step involves object detection, a challenging problem that requires methods for "(e.g. where are they), object localization (e.g. what are their extent), and object classification (e.g. what are they)" [4]. To do so, I have chosen to use CCN.  

A Convolution Neural Network (R-CNN) is a machine learning algorithm that helps machines understand image features with foresigts and remember these features to further guess whether the name of the new image is fed to the machine. 

I have separated the data into two sets (training and testing) and further aim to predict if a new image contains trees or not. Such is fed to the encoder as:
- has tree : return [1, 0]
- doesn't have tree : return [0, 1] 

I have set the learning rate (LR) to 0.001, a commonly used value for deep learning models as small learning rates are prone to having slower covergence while large learning rates are tend to cayse instability during the training process. 

***

### Implementation:

Trees vary in shape, color, density, and foliage, which can become a struggle for AI to determine a pattern as the model will have to be trained to recognize intra-class variations (different species of trees), scale variations (trees have different sizes), perspective variation (images portray trees from different angles), different illumination levels (influenced by different stages of the day such as morning, dusk, or night), and background clutter, to name a few. Due to all these factors, I highly based myself on a research paper [1] on tree species classification to design an architecture that best suits the task.

To better train my module, I collected data (520 images) from two different urban environments with different biomes and cultural settings, thus having two cities which will provide enough diversity in terms of tree species, scale, perspective, illumination, and clutter, strengthening my model to more accurate and precise levels.

I chose to work with rather small datasets (training = 260, testing = 260, validating = a combination of 260 random images from both training and testing sets) as this will result in noisy updates to model weights; that is to say, that there will be many updates with different estimates of the gradient error. This can be useful, resulting in faster learning and (sometimes) a more robust model as noisier batch sizes offer better regularizing effects, lower generalization error, and make it easier to fit one batch worth of training data in GPU memory.

In preparing the image data, I ensured to reshape the images into a 3D NumPy array with dimensions (IMG_SIZE, IMG_SIZE, 1). This is necessary because the input data for many machine learning models, particularly image-based models, are typically 3D arrays with dimensions (height, width, channels). The third dimension (here, with a value of 1) represents the number of color channels in the image data. Since the images are grayscale, there is only one channel. By reshaping the image data in this way, it becomes compatible with many machine learning frameworks and can be easily passed into a model for training or inference. Additionally, by assigning the reshaped data to a new variable ("data"), the original image data can be preserved for comparison or other purposes.

My solution was an implementation of VGG16 as it has a simple and straightforward architecture (13 convolutional layers and 3 fully connected layers), has state-of-the-art performance in many image datasets, has been pre-trained on the ImageNet dataset which means it can be used as a starting point for training new image classification models (aka transfer learning which significantly reduces the amount of data and time needed to train a model from scratch), and because this CNN highly benefits from its open-source model that is widely used for popular deep learning frameworks such as TensorFlow which I used to implement my code.  

***  

### Results:  

I ran each model once with 10 epochs, achieve an average of 63% in accuracy for City A and an average of 78% for City B. In cross-scenario examples, City A’s model on City B’s dataset performed better than City B’s model on City A’s data: this was expected as City A has a wider variety of flora (more diverse tree species, different colour and shape in foliage) as compared to City B which has more similarly looking trees, thus limiting what City B’s model perceived as a tree. Nonetheless, City B’s model performed better in classifying images in the “not” label (does not have tree in image), which I believe to be due to City B’s tree species being more similar which allows the module to grow more robust when it comes to learning to identify an object as a tree. 

***

### Conclusion:

The models worked satisfyingly with an average of 60-80% accuracy (depending on the 
model and if it was being used on its city or on the other city). The model for City A performed better 
in the cross scenario example as the tree species from City A are similar to the majority of species from 
City B, yet the module from City B performed better in both scenarios when it came to label images 
as containing a tree or not when compared to how City A’s module analysed this. 

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

[7] “Python: Image classification using keras,” GeeksforGeeks, 03-Feb-2023. [Online]. Available: https://www.geeksforgeeks.org/python-image-classification-using-keras/. [Accessed: 25-Mar-2023]. 

***

### License:  

This project follows MIT Lience - see [LICENSE](https://github.com/westrany/CSCU9M6_Tree-Classification-in-Ponta-Delgada-vs-Stirling/blob/main/LICENSE) for more details. Due to the library wraps used, other free license types might be inherited.
