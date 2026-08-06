# Bank Customer Churn Prediction Using Machine Learning

## TechCrush Cohort 7 — Group 9 Capstone Project

This project develops a machine-learning proof of concept for identifying bank customers who may be at risk of churning within the next six months.

It uses a synthetic Nigerian bank-customer dataset containing demographic, financial, transactional, engagement, service-experience and account-behaviour variables. The aim is to demonstrate how predicted churn risk can support customer-retention prioritisation.

## Project Objectives

* Explore customer characteristics and behaviours associated with churn.
* Prepare numerical and categorical data for machine learning.
* Train and compare multiple classification models.
* Evaluate performance using metrics suitable for an imbalanced target.
* Identify influential churn predictors.
* Group customers into actionable risk tiers.
* Export the verified model for interface development.

## Live Application

The trained churn prediction model is available as an interactive Streamlit application:

[Launch the Bank Customer Churn Risk Assessment App](https://group9c7financialchurnproject-gfwaeybwfqin5sodevpswe.streamlit.app/)

The application accepts the 45 customer inputs used by the model and produces:

- Predicted churn probability
- Classification decision using the selected 0.57 threshold
- Customer risk tier
- Human-supervised retention recommendation

> **Important:** This application is an educational proof of concept trained on synthetic data. It should not be used for real banking decisions without retraining and validation using genuine customer data.

## Dataset

The project uses a fully synthetic dataset created for academic and educational purposes. It contains no genuine bank records, personally identifiable information or confidential customer data.

| Item                 |             Value |
| -------------------- | ----------------: |
| Customer records     |            15,000 |
| Total columns        |                48 |
| Model input features |                45 |
| Duplicate rows       |                 0 |
| Missing values       |                 0 |
| Target variable      | `Churn_Within_6M` |

### Target Distribution

| Churn status | Customers | Percentage |
| ------------ | --------: | ---------: |
| No           |    12,251 |     81.67% |
| Yes          |     2,749 |     18.33% |

Because the target is imbalanced, model selection was not based on accuracy alone. Churn precision, recall, F1 score, balanced accuracy and the confusion matrix were also considered.

## Project Workflow

1. Data loading and quality checks
2. Exploratory data analysis
3. Feature preparation and preprocessing
4. Training and held-out test splitting
5. Model training and comparison
6. Cross-validation
7. Classification-threshold selection
8. Final test evaluation
9. Feature-importance and error analysis
10. Customer risk-tier development
11. Model verification and export

The data was divided into:

* **Training and development set:** 12,000 customers
* **Held-out test set:** 3,000 customers

The held-out test set was not used during model development or threshold selection.

## Models Evaluated

* Dummy Classifier
* Logistic Regression
* Random Forest
* Gradient Boosting

The final selected model was a **weighted Gradient Boosting pipeline** using a fixed classification threshold of **0.57**.

The threshold was selected using out-of-fold predictions from the training and development data before the final model was evaluated on the held-out test set.

## Final Model Results

| Metric             | Result |
| ------------------ | -----: |
| Test customers     |  3,000 |
| Accuracy           | 79.77% |
| Churn precision    | 45.47% |
| Churn recall       | 52.00% |
| Churn F1 score     | 48.52% |
| Balanced accuracy  | 69.00% |
| Customers flagged  |    629 |
| Percentage flagged | 20.97% |

### Confusion Matrix Summary

| Outcome         | Customers |
| --------------- | --------: |
| True negatives  |     2,107 |
| False positives |       343 |
| False negatives |       264 |
| True positives  |       286 |

The model identified **286 of the 550 actual churners**, giving a churn recall of **52.00%**, while flagging **20.97% of the test population** for possible retention attention.

This result is more meaningful than accuracy alone because a majority-class prediction would already achieve relatively high accuracy on the imbalanced dataset.

## Customer Risk Tiers

| Risk tier | Predicted probability | Observed churn rate |
| --------- | --------------------: | ------------------: |
| Low       |            Below 0.30 |               6.18% |
| Moderate  |    0.30 to below 0.57 |              13.78% |
| High      |    0.57 to below 0.75 |              33.63% |
| Very High |        0.75 and above |              58.78% |

The risk tiers are intended to help prioritise customer review. They should not automatically determine customer treatment without human judgement and additional customer information.

## Key Predictive Areas

Feature-importance analysis showed that influential predictors included variables relating to:

* Customer inactivity
* Active-member status
* Recent transaction decline
* Number of products held
* Account fees relative to customer income

These are predictive associations in the synthetic dataset and should not be interpreted as proven causes of churn.

## Repository Structure

```text
├── data/
├── documentation/
├── notebooks/
│   └── Bank_Customer_Churn_Prediction_Final.ipynb
├── outputs/
├── presentation/
├── README.md
└── requirements.txt
```

The complete analysis is available in:

[`notebooks/Bank_Customer_Churn_Prediction_Final.ipynb`](notebooks/Bank_Customer_Churn_Prediction_Final.ipynb)

## Running the Notebook

1. Open the final notebook from the `notebooks/` folder.
2. Select **Open in Colab**.
3. Confirm that the dataset path points to the repository data file.
4. Install any required packages if prompted.
5. Select **Runtime → Restart session and run all**.
6. Allow the cells to run sequentially.

The final notebook was restarted and executed from beginning to end without errors.

## Model Export and Interface

The fitted preprocessing-and-model pipeline was exported with:

* The fixed `0.57` classification threshold
* The required 45-feature schema
* Risk-tier boundaries
* Model metadata and software versions
* A sample input and expected prediction

The exported package was reloaded successfully and reproduced the notebook predictions.

A separate interface is planned to collect customer information and display the predicted churn probability, churn status, risk tier and suggested retention attention.

## Limitations and Future Improvements

This project is an educational proof of concept and has the following limitations:

- The model was trained on synthetic customer data. Its performance therefore does not establish that it will generalize to genuine bank customers.
- The full model uses 45 input features. Although this allows the model to evaluate a broad range of customer characteristics, manually entering all 45 values may reduce the usability of the current standalone application.
- The inclusion of 45 features does not mean that every feature contributes equally to the prediction.
- Some demographic and geographic relationships learned from synthetic data may not represent genuine customer behaviour and should not be interpreted as causal relationships.
- The application provides decision support only and should not be used as the sole basis for real customer-retention decisions.

Future development should compare the full model with reduced-feature models using feature-importance and feature-selection methods. Models using the strongest 10, 15 or 20 predictors can be evaluated to identify the smallest feature set that maintains acceptable predictive performance. In a real banking environment, many required values would also be retrieved automatically from the bank’s database rather than entered manually.

## AI-Assisted Development Disclosure

OpenAI ChatGPT supported synthetic-data development and validation, analytical planning, portions of code development and refinement, debugging, interpretation and documentation.

The project team executed and reviewed the workflow and remains responsible for the submitted code, results, conclusions and recommendations. No genuine or confidential bank-customer information was provided to the AI tool.

## Project Status

* [x] Data preparation and validation
* [x] Exploratory data analysis
* [x] Model development and comparison
* [x] Threshold selection
* [x] Held-out test evaluation
* [x] Risk-tier development
* [x] Final notebook verification
* [x] Model export and reload verification
* [ ] User-interface development
* [ ] Interface testing and deployment
* [ ] Final presentation preparation

## Conclusion

The project demonstrates how machine learning can help prioritise bank customers with elevated predicted churn risk.

The selected weighted Gradient Boosting model identified **52.00% of actual held-out churners while flagging 20.97% of customers**. Although the result provides useful prioritisation, genuine banking data and further validation would be required before operational use.
