# Tree Image Classification with Deep Learning

_Note: the following project is an academic assignment part of my Data Science Bachelor's at the University of Stirling._

### Introduction:
***

Classification is a machine learning model used to predict mutually exclusive categories when there are no continuous values. It does so by using labels to classify data, thus predicting a discrete number of values.  

Deep Learning is one of many of machine learning methods which bases itself on artificial neural networks with multiple layers of processing that are used to extract progressively higher level features from data; such can be supervised, semi-supervised or unsupervised.  

This Contextual Image Classification Project makes use of Deep Learning to train a Classification model to recognize and correctly classify trees in an urban context. Deep Learning models often require a considerable amount of relevant training data to be able to provide good results, thus the collection of 400 eye-picked images that depict trees in different settings, seasons and lightings. When compiling this dataset, I had extra care to ensure the images I was collecting were either public domain or free to use as long as the owner is ackowledged; to ensure this, I created a csv file with the sources for each image. 



### Dataset:
***

Trees vary in shape, colour, density and folliage, which can become a struggle for AI to determine a pattern as the model will have to be trained to recognize intra-class variations (different species of trees), scale variations (trees have different sizes), perspective variation (images portray trees from different angles), different illumination levels (influenced by different stages of the day such as morning, dusk, or night), and background cluttler, to name a few.  

To better train my module on this, I chose to collect image data from two different urban environments with different biomes and cultural settings, thus having two cities (A and B, with 200 images each) which will provide enough diversity in terms of tree species, scale, perspective, illumination and clutter, strengthening my model to more accurate and precise levels.  

• City A is my birthtown, Ponta Delgada (São Miguel, Azores, PT), with an Oceanic climate and Azorean Macaronesian Flora (also refered to as the Laurissilva Forest) with over 60 unique plants (special emphasis on Cedar (_Arceuthobium azoricum_), "Pau Branco" (_Picconia azorica_) and Laurel tree (_Laurus azorica_)).  

• City B is my current city of residence, Stirling (Scotland, UK) as well with an Oceanic climate but with a different Flora constituted by heather moorland, coastal machair and a reduced boreal Caledonian forest. The similarity in climate (both cities have Oceanic climate) can benefit the AI model as tree species will have some similarities in vegetation; their different Floras, however, can complicate the classification problem as the AI will have to learn to identify more varieties of trees, which in itself is a perk as it will train the module to recognize a broader scope of tree species. 



### Implementation: 
***
[to be added]  



### Conclusion:
***
[to be added]



#### File Index:  
***

- [CityA_images](https://github.com/westrany/CSCU9M6_Tree-Classification-in-Ponta-Delgada-vs-Stirling/tree/main/CityA_images): 200 image database from City A
- [ImageSource.csv](https://github.com/westrany/CSCU9M6_Tree-Classification-in-Ponta-Delgada-vs-Stirling/blob/main/ImageSource.csv): a database with image sources  
- [2023_Spring_Assignment_M6.ipynb](https://github.com/westrany/CSCU9M6_Tree-Classification-in-Ponta-Delgada-vs-Stirling/blob/main/2023_Spring_Assignment_M6.ipynb): main Jupyter Notebook with project instructions and assembled code



#### License:  
***

This project follows MIT Lience - see [LICENSE](https://github.com/westrany/CSCU9M6_Tree-Classification-in-Ponta-Delgada-vs-Stirling/blob/main/LICENSE) for more details. Due to the library wraps used, other free license types might be inherited.
