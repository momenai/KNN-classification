 ## About the problem
 I have implemented a nearest neighbor algorithm from scratch to classify glass types of different glass . I have also extended the implementation as weighted KNN as well. 

Dataset consists of 214 samples with discrete 6 class types. (Type 4 class have no samples in the dataset, thus the dataset includes 6 different class types {”1”, ”2”, ”3”, ”5”, ”6”, ”7”}

## Attribute Information:
1. RI: refractive index
2. Na: Sodium
3. Mg: Magnesium
4. Al: Aluminum
5. Si: Silicon
6. K: Potassium
7. Ca: Calcium
8. Ba: Barium
9. Fe: Iron
10. Type: Type of glass: (class attribute)

## Learning about: 
1. Feature Normalization
2. Shuffle all data using numpy permutations
3. Cross validation for dataset
4. Implemented a nearest neighbor algorithm
5. Implemented weighted nearest neighbor algorithm 

## Observations and result analysis
1. Selecting K as 1 or 3, is the most convientient for this tasks. 
2. weighted_normalized data give in avarage the 67% as accuracy
3. The more datat we have, the more the accuracy improved. 
4. When we have less Data, crross valiation is important. 
5. Without shuffling data,  accuracy goes down. KNN does not generalize well. 
6. Learning happended, we we shuffle data and diverse the features. 
7. Accuracy may be improved by adding more data. 
8. Cross validation increased time of training better to now use 5 K. 3 maybe good. because I does not increaze overall accuracy that much. 
9. "Weighted non-normalized KNN"is the right implmentation for glass classification.
10. Weighted Normalized KNN is not a good choice for this problem. 
11. Normalization data does not help much when the distance are too close 
