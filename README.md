# **ACM Research Coding Challenge (Spring 2023 - Shahrukh Showkath**

## Introduction to Approach

Upon inspecting the dataset, I instantly recognized this problem to be a classification program as the data of the stars can be used to predict what type of star it is classified as. With this in mind, I did some research on how these stars are exactly classified (https://en.wikipedia.org/wiki/Stellar_classification). The article gave me a quick overview into the Harvard spectral classification system that uses general data about the stars for classification; for example, a blue star that has an effective tempearture over 30,000 K would be a class O star, or Blue Supergiant. This information is important as it reveals that the star's classification comes from a series of simple inputs and is on a general scale. This was furthered when looking at the Hertzsprung-Russell Diagram which shows general linear trends between a star's numerical characteristics and its classification. From here, I decided that a linear regression machine learning model would be a good approach as the data seemed to be generally linear in nature. I additionally decided to use a K-Nearest Neighbors(KNN) model as well to simply compare it with another classification approach.

## Approach

I first conducted exploratory data anlaysis on the dataset. I checked what the data looked like, the data types involved, and the number of rows/columns the dataset had. I noticed that some of the data was numerical and others were categorical. I used one hot encoding on the categorical data to standardize the data types throughout. I additionally used a pearson correlation matrix to determine if there were any strong correlations between the features. The strongest correlations with star type included Luminosity, Radius, and Magnitude. I decided to train two seperate models: one where all the features were given and another with just these three features. The reasoning for this decision is to possibly account for overfitting by giving the model too many features and therfore only use the most relevant features. With the datasets cleaned, I used the scikit-learn library to access their linear regression and KNN models and trained the relevant datasets. I set aside 80% of the dataset to train the models and reserved the other 20% to test the model. I used an arbitrary 7 neighbors for the KNN model.


## Results and Analysis

The linear regression model with only luminosity, radius, and magnitude as features performed well, with about `91%` accuracy. The model with all the features performed about 5% better, with around `96%` accuracy. These are very excellent accuracy scores and it can be easily attributed to the fact that the categories of the stars are based on continuous inputs, such as temperature and size. However, the KNN models performed worse at about `50%` and `60%` accuracy between the model with limited features and all features.

I felt that linear regression was the more intuitive approach for this problem and performed well in classification to boot.

## Conclusion
In essence, a linear regression model was able to decently classify what type of star the data was given its characteristics. This makes sense as seen with the Hertzsprung-Russell Diagram that shows a general trend between a star's classification and its numerical characteristics like heat and size. The KNN model did not perform as well which can be attributed to possible overfitting by giving it too many features or data imbalances as there were not an equal number of star types. The main limitation of this approach is that if if given more data, the model may be prone to outliers. As seen in the provided diagrams from Kaggle, there is a lot of overlap between the star types and their characteristics. The regression model relies on there to be a direct linear correlation between the values which may not always be fulfilled when accounting for more stars. 

However, with the given dataset, the program is able to provide a relatively accurate classification of stars.

## Libraries

- numpy
- pandas
- scikit-learn


## Sources
- [Stellar classification](https://en.wikipedia.org/wiki/Stellar_classification)
- [sklearn Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [sklearn KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Star Kaggle Dataset](https://www.kaggle.com/datasets/deepu1109/star-dataset)
