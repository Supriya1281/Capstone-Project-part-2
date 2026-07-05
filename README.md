# Capstone-Project-part-2
Supervised Machine Learning Model — Build, Train, and Evaluate

With a cleaned dataset in hand, I produced two predictive models: one for regression (predicting a continuous output) and one for binary classification. I preprocessed the data correctly to avoid data leakage, handle class imbalance where applicable, and evaluate both models rigorously.

 Starting with loading the cleaned dataset which was previously cleaned in part1 The Machine learning predictive models are as follows:

1. Dataset Labels & Feature Matrix

To evaluate both continuous and discrete predictive models, our target variables are separated into two distinct labels:

*   **Regression Label (`y_reg`)**: Defined as the `TotalCharges` column. This is a continuous numeric variable representing the cumulative billed amount per customer over their tenure. It is isolated for regression-based machine learning tasks.
*   **Classification Label (`y_clf`)**: Defined using the natural binary column `Churn`. The categorical values ('yes' / 'no') have been mapped to an integer boolean format (1 / 0). This label represents whether a customer terminated their service and is used for binary classification tasks. 
    * *Alternative*: A classification label can also be derived by binarizing the regression label at its median (`y_clf = (y_reg > y_reg.median()).astype(int)`). For `TotalCharges`, the dataset median is roughly **1,897.93**.
*   **Feature Matrix (`X`)**: Contains all remaining descriptive customer attributes (9 features in total). The target variables (`TotalCharges` and `Churn`) alongside the non-predictive `customerID` string have been explicitly excluded to prevent data leakage during model training. Feature inputs include demographics, service types, and account durations.


2.Categorical Data Encoding

To prepare our non-numeric features for machine learning algorithms, we applied two distinct encoding strategies based on the nature of the data:

1. Label Encoding (Ordinal Features)
We applied label encoding to the `Contract` feature by mapping its string values to integers based on a predefined dictionary (`month-to-month: 0`, `one year: 1`, `two year: 2`). 
*   **Justification:** The contract duration possesses a distinct natural order (shortest duration to longest). Mapping these values to an increasing integer sequence preserves this monotonic relationship. Predictive models can leverage this ordinality (e.g., recognizing that a two-year contract implies a higher commitment level than a one-year contract).

2. One-Hot Encoding (Nominal Features)
We applied one-hot encoding to features without a natural hierarchy (`gender`, `PhoneService`, `InternetService`, `PaperlessBilling`, `PaymentMethod`) using `pd.get_dummies()`. To prevent multicollinearity (the dummy variable trap), we explicitly dropped the first dummy column for each feature (`drop_first=True`).
*   **Why One-Hot Encoding?** If we were to apply label encoding to nominal features (e.g., mapping `PaymentMethod` to 0, 1, 2, 3), the model's underlying mathematical equations might misinterpret these arbitrary integers as having a false ordinal relationship. For instance, the model might incorrectly assume that an "electronic check" (3) is "greater than" or "worth three times as much as" a "credit card" (1). One-hot encoding avoids this by creating isolated binary features for each category, completely neutralizing the risk of false mathematical ranking.

3.Train-Test Split & Feature Scaling

Train-Test Split
To evaluate our models on unseen data, we partitioned the dataset using an 80/20 split (`test_size=0.2`), reserving 80% of the data for training the model and 20% for testing. We used a fixed random state (`random_state=42`) to ensure reproducibility across different runs. The feature matrix (`X`) and both target labels (`y_reg` and `y_clf`) were split simultaneously to maintain perfectly aligned indices.

Feature Scaling & Preventing Data Leakage
Machine learning algorithms generally perform better (and converge faster) when numerical features are on a similar scale. To achieve this, we applied Standardization (`StandardScaler`), which transforms the data so that it has a mean of 0 and a standard deviation of 1.

**CRITICAL NOTE ON DATA LEAKAGE:** 
The `StandardScaler` was explicitly fit **only on the training features** (`scaler.fit(X_train)`). The resulting fitted scaler was then used to transform both the training set and the test set (`scaler.transform(...)`). 

Fitting the scaler on the *entire* dataset before splitting would constitute **data leakage**. If the scaler sees the full dataset, it encodes the global statistics (mean and variance) of the test set into the transformation parameters. Consequently, information about the test set "leaks" into the training process, giving the model an unfair advantage and resulting in artificially inflated, unreliable performance metrics. By fitting only on the training set, we strictly treat the test set as entirely unknown, future data.

4.Regression Modeling: OLS vs Ridge

To predict the continuous target `TotalCharges`, we evaluated two regression models: Ordinary Least Squares (OLS) Linear Regression and Ridge Regression.

Interpreting Model Coefficients

Because our feature matrix was standard-scaled prior to training, the coefficients represent the predicted change in `TotalCharges` for every **one standard deviation increase** in that feature, holding all other features constant.
*   **Large Positive Coefficient:** Implies a strong direct relationship. For example, a one standard deviation increase in `tenure` (our strongest positive driver) is associated with an increase of roughly $1,117.45 in the predicted `TotalCharges`.
*   **Large Negative Coefficient:** Implies a strong inverse relationship. For example, the negative coefficient for `PaymentMethod_credit card` (-$84.95) indicates that if a customer uses this payment method, the model predicts their total charges will decrease by roughly $84.95 compared to the baseline, assuming all other variables remain identical.

OLS vs. Ridge Performance Comparison

| Model | Mean Squared Error (MSE) | R² Score |
| **Linear Regression (OLS)** | 1521930.38 | 0.3931 |
| **Ridge Regression (alpha=1.0)** | 1521625.29 | 0.3932 |

 Why Use Ridge Regression?
While Standard OLS Linear Regression seeks to minimize only the residual sum of squares (the difference between predicted and actual values), Ridge Regression might produce a different coefficient profile because it incorporates an L2 penalty term into its cost function. The `alpha` parameter controls the strength of this penalty. A higher `alpha` applies stronger regularization, forcing the model to shrink the coefficients of less important features closer to zero. This shrinkage helps mitigate multicollinearity (when features are highly correlated with each other) and reduces model variance, which often helps prevent overfitting and improves generalization on unseen data—as slightly reflected by Ridge's lower MSE in our evaluation.

5.Classification Modeling: Logistic Regression

To predict whether a customer will churn (`y_clf`), we trained a Logistic Regression model (`max_iter=1000`).

Handling Class Imbalance
During exploratory analysis, we identified a mild class imbalance: the minority class (Churn = yes) represented only 33.5% of the training dataset. To prevent the model from becoming biased toward the majority class (predicting 'no churn' by default), we utilized the **`class_weight='balanced'`** hyperparameter in our Logistic Regression constructor rather than applying a synthetic sampling technique like SMOTE. 
*   **Why `class_weight='balanced'`?** This approach modifies the underlying loss function directly. It assigns a higher mathematical penalty to misclassifying the minority class, inversely proportional to the class frequencies. It avoids the computational overhead and potential noise introduced by synthetic data generation while ensuring both classes are treated fairly during gradient descent.

 Evaluation Metrics

To evaluate model performance, we look at several metrics based on True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN):

*   **Precision:** Represents the accuracy of our positive predictions. Out of all the customers the model predicted would churn, how many actually did?
     \text{Precision} = \frac{TP}{TP + FP} 

*   **Recall:** Represents our model's ability to identify all actual positive instances. Out of all the customers who *actually* churned, how many did the model successfully flag?
     \text{Recall} = \frac{TP}{TP + FN} 

**Which metric is more important?**
For this specific task (Churn Prediction), **Recall** is generally considered the more critical metric. A False Negative (the model predicts a customer will stay, but they actually leave) means losing a customer and their revenue entirely. A False Positive (the model predicts a customer will leave, but they were going to stay anyway) simply means we might unnecessarily spend resources on a retention campaign or discount for that customer. Because the cost of losing a customer usually outweighs the cost of a retention email, maximizing our True Positive rate (Recall) is the priority. Our tuned model achieved an impressive **84% Recall** for the churn class.

 ROC Curve and AUC
To visualize the tradeoff between the True Positive Rate and the False Positive Rate across different classification thresholds, we generated a Receiver Operating Characteristic (ROC) curve. 

Our model achieved an **Area Under the Curve (AUC) of 0.713**.
*   **What this means:** The AUC value represents the probability that the model will rank a randomly chosen positive instance (a churner) higher than a randomly chosen negative instance (a non-churner). An AUC of 0.713 indicates that our model has a fair-to-good ability to successfully separate and distinguish between the two classes, performing significantly better than a random guess (which would yield an AUC of 0.5).
*   <img width="691" height="545" alt="AUC" src="https://github.com/user-attachments/assets/8022e4f3-ad14-4adc-b657-024c4c5be0db" />
     AUC Curve Image

6.Decision Threshold Sensitivity Analysis

By default, Logistic Regression assigns a positive class label if the predicted probability is $\ge 0.50$. However, this threshold can be tuned to optimize for different business objectives. We tested probability thresholds from 0.30 to 0.70 to observe the trade-offs between precision and recall.

Core Metric Formulas
Our evaluation relies on the following metrics, defined by True Positives (TP), False Positives (FP), and False Negatives (FN):
*   **Precision:** {TP}/{TP + FP}
*   **Recall:** {TP}/{TP + FN}

Optimal F1-Score
The F1-score is the harmonic mean of Precision and Recall. Based on our sensitivity analysis, a **threshold of 0.50** maximizes the F1-score on this dataset (F1 = 0.5185), closely followed by 0.40 (F1 = 0.5181).

Business Context: Precision vs. Recall Trade-off
In the specific context of **Churn Prediction**, missing a customer who is about to leave (False Negative) results in lost recurring revenue, which is generally much more expensive than mistakenly targeting a loyal customer with a retention email (False Positive). Therefore, **Recall is the more important metric** for this task. 

*   **How to optimize:** To maximize Recall (capture as many potential churners as possible), we would **lower the decision threshold** (e.g., to 0.30 or 0.40). 
*   **The Cost:** Lowering the threshold inherently lowers Precision. The model becomes hyper-sensitive and flags many customers who were never going to churn. The business cost of this adjustment is the financial overhead of offering unnecessary discounts, retention incentives, or marketing resources to False Positives.

7.Regularization Experiment: Logistic Regression

To understand the impact of model complexity on our churn predictions, we conducted an experiment adjusting the regularization strength of our Logistic Regression model. We compared our baseline model against a heavily regularized variant. 

Performance Comparison

| Model Variant | Precision | Recall | AUC |
| **Baseline (C = 1.0)** | 0.3750 | 0.8400 | 0.7133 |
| **Strong Regularization (C = 0.01)** | 0.3750 | 0.8400 | 0.7130 |

The Role of 'C' in Logistic Regression

In Scikit-Learn's Logistic Regression, the `C` parameter represents the **inverse of regularization strength** (similar to the support vector machine parameter). A smaller `C` value specifies stronger regularization, meaning the model applies a heavier L2 penalty to the objective function, forcibly shrinking the model's coefficients closer to zero to prevent overfitting. 

**Impact on this dataset:** 
Reducing `C` to 0.01 resulted in practically identical performance on this dataset. The Precision and Recall remained exactly the same at the default 0.5 decision threshold, while the AUC experienced a negligible decrease (from 0.7133 to 0.7130). This indicates that our baseline model (with only 12 features) was not suffering from high variance or severe overfitting to begin with. Because the feature space is relatively small and straightforward, forcing stronger coefficient shrinkage via a lower `C` did not yield any improvements in generalization, and slightly worsened the model's overall discriminative ability across all thresholds (AUC). Therefore, the baseline `C=1.0` model remains the optimal choice.

8.Statistical Reliability of Model Performance

In our regularization experiment, the baseline Logistic Regression model (`C=1.0`) achieved an AUC of 0.7133, while the strongly regularized model (`C=0.01`) achieved an AUC of 0.7130. To quantify how reliably the `C=1.0` model outperforms the `C=0.01` model, we conducted a bootstrap hypothesis test.

Methodology
We drew **$n=500$** bootstrap samples from the test set. For each iteration, we randomly sampled rows with replacement, isolated the corresponding true labels and predicted probabilities for both models, and calculated the AUC difference (AUC of C=1.0 minus AUC of C=0.01). 

Results
After 500 iterations, we computed the mean difference and the 95% Confidence Interval (bounded by the 2.5th and 97.5th percentiles):
*   **Mean AUC Difference:** 0.00023
*   **2.5th Percentile (Lower Bound):** -0.01172
*   **97.5th Percentile (Upper Bound):** 0.01180

Conclusion
The 95% confidence interval for the AUC difference ranges from approximately -0.0117 to +0.0118. **Because this interval includes zero, the negligible difference in performance between the two models is not statistically reliable.** We cannot confidently claim that the `C=1.0` model will consistently outperform the `C=0.01` model on new, unseen data samples. The slight variation observed in our initial test set evaluation is likely due to random sampling noise rather than a meaningful difference in the models' underlying discriminative power.









