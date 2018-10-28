# Team7_Project2
Team 7 Repository for the second project (Image retrieval in the museum dataset)

## Requirements

The file ```requirements.txt``` contains all the required packages to run our code. This project has been developed in Python 3.7.

In order to run our code, you need to place the ```museum_set_random``` , the ```query_devel_random```  and the ````query_test_random```` folders in the root folder of this project.

## Running the code
We have implemented two main methods, one using histogram based matching and one using Discrete Wavelet Transformation based hashing. We have also combined both in order to increase the scores. 

### Histogram based matching

The histogram based method uses a spatial pyramidal representation of the images in order to compare them. We use the Bhattacharyya distance to compare the histograms.

In order to execute this method on the validation query set run ````python retrieve_img_1.py````.
 
To run this method on the test queries run ````python retrieve_img_1.py -test````. 

### DWT based hashing matching

This method uses DWT based hashing to compare the images. This hashing method scales the images to a certain size and computes their hash using the discrete wavelet transformation. We calculate the distance of the hashes in order to compute the difference between two images. 

In order to execute this method run ````python retrieve_img_2.py````.

To run this method on the test queries run ````python retrieve_img_2.py -test````.

We based our code on [this](https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5) article.

### Histogram matching +  hashing matching

This method uses the previous two methods to match the images. In this method we add the two previous scores for every pair of images. 

In order to execute this method run ````python retrieve_img_2.py  -use_histogram````.

To run this method on the test queries run ````python retrieve_img_2.py  -use_histogram -test````.

## Results

Those are the results obtained with the different matching methods (Intel Core I7 2600K @3.40GHz). The number in the hashing method is the size of the hash: 

Method | Histogram based| DWT hash 4 | DWT hash 8 | DWT hash 16 | DWT hash 32 | Hash 16 + Histogram
--- | --- | --- | --- |--- |--- |---  
Score | 0.95 | 0.52 | 0.85 | 0.92 | 0.92 | 1.0
Score Test | 0.81 | 0.54 | 0.81 | 0.88 | 0.90 | 0.93 
Time (in seconds) |5|12|12|12|12|\>14|


