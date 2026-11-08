# Dry Bean Classification

## Problem Statement

In this project, I classify dry beans into 7 different types. The model uses only the shape of each bean, like its area, perimeter and roundness. I trained 5 machine learning models and compared them to see which one works best.

## Dataset Description

- **Name:** Dry Bean Dataset (UCI Machine Learning Repository)
- **Source:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- **Rows:** 13,611
- **Features:** 16 shape features (Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4)
- **Classes:** 7 (DERMASON, SIRA, SEKER, HOROZ, CALI, BARBUNYA, BOMBAY)
- **Missing values:** 0
- **Class imbalance:** Some classes have many more beans than others. DERMASON has 3,546 beans, but BOMBAY has only 522. So I used a stratified split, weighted averages for precision, recall and F1, and also MCC, because MCC works well when classes are not balanced.

## GitHub Repository Link

https://github.com/princeroy2593/dry-bean-classifier

## Streamlit App Link

https://prince-roy-ml-assignment.streamlit.app

To test the app, download test_data.csv from this repository and upload it.

## Models Used

All scores are on the test set (20% of the data).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.921 | 0.993 | 0.922 | 0.921 | 0.922 | 0.905 |
| Decision Tree | 0.892 | 0.933 | 0.892 | 0.892 | 0.892 | 0.870 |
| KNN | 0.917 | 0.981 | 0.917 | 0.917 | 0.917 | 0.899 |
| Naive Bayes | 0.898 | 0.990 | 0.901 | 0.898 | 0.898 | 0.877 |
| SVM | 0.922 | 0.994 | 0.922 | 0.922 | 0.922 | 0.906 |

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression did much better than I expected for such a simple model. It got 0.921 accuracy and 0.993 AUC, which is almost the same as SVM. Like the other models, its main problem was DERMASON and SIRA, as it predicted 52 DERMASON beans as SIRA and 43 SIRA beans as DERMASON. Since a linear model works this well, I think most of the bean types can be separated quite easily once the features are scaled. |
| Decision Tree | Decision Tree was the weakest model, with the lowest value in all 6 metrics (accuracy 0.892, MCC 0.870). Its AUC was only 0.933, while all the other models were above 0.98. It struggled the most with SIRA beans and got only 415 out of 527 correct, mostly because it called 69 of them DERMASON. A single tree can overfit the training data, so it does not do as well on new data. |
| KNN | KNN gave good results with 0.917 accuracy and 0.899 MCC, so it came third. Scaling was important here because KNN uses distance, and features like Area are much bigger than features like roundness. It had the same DERMASON and SIRA problem (53 and 51 mistakes) and also confused 21 BARBUNYA beans with CALI. Its AUC (0.981) was a bit lower than Logistic Regression and SVM. |
| Naive Bayes | Naive Bayes got 0.898 accuracy, which is only a little better than Decision Tree. Its two biggest mistakes were 40 BARBUNYA beans predicted as CALI and 78 DERMASON beans predicted as SIRA. I think this is because Naive Bayes assumes all features are independent, but in this dataset many features are related, for example Area, Perimeter and ConvexArea. Still, its AUC was high (0.990), so its probabilities ranked the classes well even when the final answer was wrong. |
| SVM | SVM gave the best results, with 0.922 accuracy, 0.994 AUC and 0.906 MCC. It got all 104 BOMBAY beans right, but every model did that, because BOMBAY beans are much bigger than the rest and easy to spot. Its biggest mistake was again DERMASON and SIRA, with 49 mistakes each way. Even so, it was the most balanced model overall, which is why its MCC was the highest. |
| Overall, Winner for your dataset? | SVM, but Logistic Regression is very close. |

## Overall Winner

**SVM** is the winner. It has the highest or equal highest score on all 6 metrics, with accuracy 0.922, AUC 0.994 and MCC 0.906. The high MCC shows it handles the unbalanced classes well. But Logistic Regression is only about 0.001 behind on most metrics, so it is also a good choice if we want a simpler and faster model.
