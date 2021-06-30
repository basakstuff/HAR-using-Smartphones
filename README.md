<h3 align="center">Human Activity Recognition Using Smartphones Data Set</h4>

 <p align="center">
 <a href="#dataset">Dataset</a> •
  <a href="#data-preprocessing">Data Preprocessing</a> •
  <a href="#models">Models</a> •
  <a href="#results">Results</a> •
  <a href="#contact">Contact</a> •
  <a href="#references">References</a> •
</p>


## Dataset

Smartphone manufacturing companies load Smartphones with various sensors to enhance the user experience. Two of the such sensors are Accelerometer and Gyroscope. Accelerometer measures acceleration while Gyroscope measures angular velocity. Experiments were carried out with a group of 30 volunteers aged 19-48 years. Each person wore a smartphone (Samsung Galaxy S II) around their waist and performed six activities (WALKING, WALKING-UPSTAIRS, WALKING-DOWNSTAIRS, SITTING-DOWN, STANDING-UP, LAYING-DOWN). Using the onboard accelerometer and gyroscope, 3-axis linear acceleration and 3-axis angular velocity were captured at a constant 50Hz rate. Experiments were video-recorded to manually label the data. The resulting dataset was randomly divided into two groups, 70% of the volunteers were selected to generate training data and 30% to generate test data.

[Dataset Link](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

## Data Preprocessing

The dataset was first checked for duplicate values. Then NaN and Null values were searched. But the dataset was fine on both counts.

![image](https://user-images.githubusercontent.com/25417307/123978375-7d12c200-d9c8-11eb-8e96-a239b264f4b7.png)

As Figure shows, almost the same number of readings were taken from all subjects, so there is no significant difference in reading.

![image](https://user-images.githubusercontent.com/25417307/123978447-88fe8400-d9c8-11eb-8fbd-6344219bceb1.png)

As seen in Figure, there are approximately the same number of data points for each movement. Assuming the participants had to walk the same number of stairs upwards as well as downwards and knowing the smartphones had a constant sampling rate, there should be the same amount of datapoints for walking upstairs and downstairs. Disregarding the possibility of flawed data, the participants seem to walk roughly 10% faster downwards. It can be said that the dataset is almost balanced.

![image](https://user-images.githubusercontent.com/25417307/123978497-93b91900-d9c8-11eb-81f3-2a66b118b9c5.png)

In Figure 3 and graph 1, it is clear that the activities are mostly separable. However, graph 2 reveals the personal information of the participants. For example, everyone has a unique/separable gait style (top right). So the smartphone should be able to detect what you are doing and also who is using the smartphone (if you are moving with it).

## Models

A machine learning model has multiple parameters that are not trained by the training set. These parameters control the accuracy of the model. For an estimator, I used a method that provides a comprehensive search on the parameter values specified. Grid search is a tuning technique that attempts to compute the optimum values of hyperparameters. It is an exhaustive search that is performed on a the specific parameter values of a model. The model is also known as an estimator.

In K-Folds Cross Validation, the data is divided into k different subsets. K-1 subsets are used to train the data and leave the last subset as test data. The average error value resulting from k experiments indicates the validity of the model. In this study, the 10-fold Cross Validation technique was used.

### 1- Logistic Regression with Grid Search

Logistic regression models the probabilities for classification problems with two possible outcomes. It's an extension of the linear regression model for classification problems (Peng et al., 2002). With the Logistic Regression classifier, I we tested the classifier with different values of Inverse of regularization strength C = [0.01, 0.1, 1, 10, 20, 30] We looked at the accuracy score on each value of C, and the change of accuracy and running time as C changes. As you can see below, with C value 20, we received the highest test accuracy of 0.9630. The best results were obtained in the L2 regularization. As can be seen in Figure 4, there are incorrect estimations in this model, especially in similar activities such as standing and sitting.

![image](https://user-images.githubusercontent.com/25417307/123978812-d67af100-d9c8-11eb-9d42-332232061bc5.png)

### 2- Linear Support Vector Classifier (Linear SVC)

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection. The Linear Support Vector Classifier (SVC) method applies a linear kernel function to perform classification and it performs well with a large number of samples (Ladicky & Torr, 2011). If we compare it with the SVC model, the Linear SVC has additional parameters such as penalty normalization which applies 'L1' or 'L2' and loss function. The kernel method can not be changed in linear SVC, because it is based on the kernel linear method. The best result in this model, 0.966, was obtained when C=1.

![image](https://user-images.githubusercontent.com/25417307/123978919-edb9de80-d9c8-11eb-83d0-57ef87651449.png)

### 3- Kernel Support Vector Machine (Kernel SVM)

Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for both classification or regression problems (Patle & Chouhan, 2013). However, SVM is mostly used in classification problems. The best acuracy value of 0.962 was obtained when C=16 and gamma=0.0078125. C tested 2.8,16 values, while gamma tested 0.0078125, 0.125, 2 values.

![image](https://user-images.githubusercontent.com/25417307/123978980-f90d0a00-d9c8-11eb-93fb-edb61a5e5ca8.png)

### 4- Decision Trees (DTs)

A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility (Quinlan, 1986). It is one way to display an algorithm that only contains conditional control statements. In this study, the depth of the decision tree was determined to increase by two from 3 to 10. The highest accuracy value of 0.873 was recorded at max_depth 9.

![image](https://user-images.githubusercontent.com/25417307/123979017-03c79f00-d9c9-11eb-9f32-dcb80ec2953d.png)

### 5- Random Forest Classifier (RF)

Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble (Breiman, 2001) . Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes model’s prediction. n_estimators is the number of trees to be used in the forest. In this model, the n_estimators value is set to increase by twenty from 10 to 201. It gives the best result with a value of 190. max_depth is set to increase by two from 3 to 15. It gave the best result as 13. The best accuracy is 0.928.

![image](https://user-images.githubusercontent.com/25417307/123979067-10e48e00-d9c9-11eb-9cd4-09ad180c637f.png)


### 6- Light Gradient Boosting Machine (Ligth GBM)

Light GBM is a gradient boosting framework that uses tree based learning algorithm. Light GBM grows tree vertically while other algorithm grows trees horizontally meaning that Light GBM grows tree leaf-wise while other algorithm grows level-wise. It will choose the leaf with max delta loss to grow (Ke et al., n.d.). When growing the same leaf, Leaf-wise algorithm can reduce more loss than a level-wise algorithm. While max_depth is selected as 1, 5, 10 and 50, n_estimators 10 is evaluated to increase by twenty in the range of 201. The best accuracy value of 0.935 was obtained when max_depth=5, n_estimators=110.

![image](https://user-images.githubusercontent.com/25417307/123979123-1c37b980-d9c9-11eb-80c0-aa800b2446f1.png)


### 7- K-Nearest Neighbors (k-NN)

The k-nearest neighbor (KNN) algorithm is one of the supervised learning algorithms that is easy to implement (Guo et al., 2004). Although it is used in solving both classification and regression problems, it is mostly used in solving classification problems in industry. In this study, the evaluation was made with the values of 3, 15, 25, 51, 101 and the best accuracy value was found to be 0.904, k=15.

![image](https://user-images.githubusercontent.com/25417307/123979185-2954a880-d9c9-11eb-96b7-2ca132270a9c.png)


## Results

The experimental project has been completed. As a result, a pre-prepared dataset related to the description of human movements with smartphones was used and the dataset selected to use was preprocessed for machine learning models. These models were Logistic Regression, Linear SVC, Kernel SVM, Decision Trees, Random Forest Classifier, Light GBM and KNN. Random sampling and cross validation methods were used in the training. The results of the experiment can be seen in Table.

| Model               | Time (minute) | Accuracy | Time/Accuracy |
|---------------------|---------------|----------|---------------|
| Logistic Regression | 12            | 0,9630   | 12,46         |
| Linear SVC          | 3             | 0,9677   | 3,10          |
| Kernel SVM          | 25            | 0,9626   | 25,97         |
| Decision Trees      | 1             | 0,8730   | 1,15          |
| Random Forest       | 46            | 0,9287   | 49,53         |
| Light GBM           | 90            | 0,9358   | 96,17         |
| KNN                 | 17            | 0,9046   | 18,79         |


As can be seen in the above results, the best accuracy value was obtained in the Linear SVC model with a result of 0.9677. Although Decision Trees gives the worst accuracy, we need to take into account the training times of all these models. The decision tree, which gave the best results when we compared the training time and the results, alone gave the worst result in accuracy. Then, Linear SVC alone gave good results in this ratio as well as in accuracy evaluation. The worst result was Random Forest, which finished in half the time but had lower accuracy, instead of the Light GBM, which had a training time of ninety minutes. As a result of all these evaluations, it was decided that the best model to be selected was Linear SVC. The parameters that provide this result were also declared in the relevant section.

## Contact

Feel free to email me at basaker98@gmail.com if you run into any issues or have questions, ideas or concerns.
Please enjoy and feel free to share your opinion, constructive criticism, or comments about my work. Thank you! 🙂

## References

* Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
* Colaboratory – Google. (n.d.). Retrieved June 11, 2021, from https://research.google.com/colaboratory/faq.html
* Confusion matrix—Scikit-learn 0.24.2 documentation. (n.d.). Retrieved June 11, 2021, from https://scikitlearn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
* Guo, G., Wang, H., Bell, D., & Bi, Y. (2004). KNN Model-Based Approach in Classification.
* Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (n.d.). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. 9.
* Ladicky, L., & Torr, P. (2011). Linear Support Vector Machines (p. 992).
* Patle, A., & Chouhan, D. S. (2013). SVM kernel functions for classification. 2013 International Conference on Advances in Technology and Engineering (ICATE), 1–9. https://doi.org/10.1109/ICAdTE.2013.6524743
* Peng, J., Lee, K., & Ingersoll, G. (2002). An Introduction to Logistic Regression Analysis and Reporting. Journal of Educational Research - J EDUC RES, 96, 3–14. https://doi.org/10.1080/00220670209598786
* Quinlan, J. R. (1986). Induction of decision trees. Machine Learning, 1(1), 81–106. https://doi.org/10.1007/BF00116251
* sklearn.metrics.classification_report—Scikit-learn 0.24.2 documentation. (n.d.). Retrieved June 11, 2021, from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html


