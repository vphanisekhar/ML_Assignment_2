\## Problem Statement

The objective of this project is to predict whether a bank customer will subscribe

to a term deposit based on demographic and campaign-related features.



\## Dataset Description

\- Dataset: Bank Marketing Dataset (UCI)

\- Rows: 41,188

\- Features: 20

\- Target: y (yes/no)



\## Models Used and Evaluation Metrics



| Model               | Accuracy | AUC    | Precision | Recall | F1     | MCC    |

| ------------------- | -------- | ------ | --------- | ------ | ------ | ------ |

| Logistic Regression | 0.9166   | 0.9424 | 0.7118    | 0.4364 | 0.5411 | 0.5162 |

| Decision Tree       | 0.8946   | 0.7327 | 0.5329    | 0.5237 | 0.5283 | 0.4690 |

| KNN                 | 0.9075   | 0.8799 | 0.6250    | 0.4472 | 0.5214 | 0.4798 |

| Naive Bayes         | 0.8203   | 0.8393 | 0.3495    | 0.6907 | 0.4642 | 0.4009 |

| Random Forest       | 0.9192   | 0.9456 | 0.7073    | 0.4817 | 0.5731 | 0.5421 |

| XGBoost             | 0.9179   | 0.9499 | 0.6595    | 0.5614 | 0.6065 | 0.5633 |



\## Observations



| Model | Observation |

|-----|------------|

| Logistic Regression | Simple baseline, decent performance |

| Decision Tree | Overfits slightly |

| KNN | Sensitive to scaling |

| Naive Bayes | Fast but assumes independence |

| Random Forest | Balanced and robust |

| XGBoost | Best overall performance |



Logistic Regression provides a strong baseline with high accuracy but lower recall,

indicating difficulty in identifying positive subscriptions.



Decision Tree shows moderate performance with lower AUC, suggesting overfitting.



KNN performance is affected by feature scaling and performs moderately well.



Naive Bayes achieves high recall but poor precision due to its independence assumption.



Random Forest provides balanced performance with strong MCC, indicating robustness.



XGBoost achieves the best overall performance across AUC, F1-score, and MCC.



