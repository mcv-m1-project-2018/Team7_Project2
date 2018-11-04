# Team7_Project2
Team 7 Repository for the second project (Image retrieval in the museum dataset)

## Requirements

The file ```requirements.txt``` contains all the required packages to run our code. This project has been developed in Python 3.7.

In order to run our code, you need to place the following folders in the root folder of this project:

Week 3: ```museum_set_random```, ```query_devel_random```, ````query_test_random````

Week 4: ````BBDD_W4````, ````query_devel_W4```` 

## Running the code - WEEK 3
We have implemented two main methods, one using histogram based matching and one using Discrete Wavelet Transformation based hashing. We have also combined both in order to increase the scores. 

### Histogram based matching

The histogram based method uses a spatial pyramidal representation of the images in order to compare them. We use the Bhattacharyya distance to compare the histograms.

In order to execute this method on the validation query set run ````python retrieve_img_1.py````.
 
To run this method on the test queries run ````python retrieve_img_1.py -test````. 

### DWT based hashing matching

This method uses DWT based hashing to compare the images. This hashing method scales the images to a certain size and 
computes their hash using the discrete wavelet transformation. We calculate the distance of the hashes in order to 
compute the difference between two images. 

In order to execute this method run ````python retrieve_img_2.py````.

To run this method on the test queries run ````python retrieve_img_2.py -test````.

We based our code on [this](https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5) article.

### Histogram matching +  hashing matching

This method uses the previous two methods to match the images. In this method we add the two previous scores for every pair of images. 

In order to execute this method run ````python retrieve_img_2.py  -use_histogram````.

To run this method on the test queries run ````python retrieve_img_2.py  -use_histogram -test````.

## Running the code - WEEK 4

For the fourth week we have used Surf, Sift and Orb to retrieve the image queries.

### SURF, SIFT and ORB

Surf, Sift and Orb are run from the ````retrieve_img_3.py```` file. To specify the feature matching method use 
````-use_surf```` to use Surf, ````-use_sift```` for Sift and ````-use_orb```` for Orb. For example, to evaluate the 
development query set with Orb run ````python retrieve_img_3.py -use_orb````.

The code runs by default on this week's development query set. If you want to run the code on this week's query set use
the ````-test```` flag. For example, run ````-python retrieve_img_3.py -test -use_surf```` to evaluate the test set with Surf. 
 
 To run the code with week's 3 development query set use the flag ````-week3````. Finally, to 
run the test set from the third week use the ````-test```` flag jointly with the ````-week3```` flag.

## Results

### Week 3

Those are the results obtained with the different matching methods (Intel Core I7 2700K @3.40GHz). The number in the hashing method is the size of the hash: 

Method | Histogram based| DWT hash 4 | DWT hash 8 | DWT hash 16 | DWT hash 32 | Hash 16 + Histogram
--- | --- | --- | --- |--- |--- |---  
Score | 0.95 | 0.52 | 0.85 | 0.92 | 0.92 | 1.0
Score Test | 0.81 | 0.54 | 0.81 | 0.88 | 0.90 | 0.93 
Time (in seconds) |5|12|12|12|12|\>14|

### Week 4

These are the results obtained with the methods used in the fourth week:

Method | Surf| Orb | Sift  
---  | --- | --- | ---  |
Score Week 4| 0.67 | 0.93 | ? |
Score Week 4 Test | ? | ? | ? | 
Score Week 3 | 0.81 | 0.98 | ? |
Score Week 3 Test | 0.77 | 0.86 | ? |
Time (in seconds) | ? | ? | ? |




