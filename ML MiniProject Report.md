## **Machine Learning Project: Regression & Classi�cation** 

## **1. Introduction** 

This project applies supervised machine learning techniques on two different datasets: 

**House Prices dataset** (regression problem) **Titanic dataset** (classi�cation problem) 

The objective is to compare multiple models, analyze their performance, and evaluate the effect of preprocessing, scaling, and model complexity. 

## **2. Datasets Description** 

## **2.1 House Prices Dataset** 

- Target variable: `SalePrice` 

- Task: Predict continuous house prices (Regression) 

- Features include both numerical and categorical attributes 

## **2.2 Titanic Dataset** 

Target variable: `Survived` 

- Task: Predict passenger survival (Classi�cation) 

- Dataset contains demographic and ticket-related features 

## **3. Data Preprocessing** 

## **3.1 Handling Missing Values** 

- Numerical features: median imputation 

- Categorical features: one-hot encoding 

Ensures models can train without errors from missing data 

## **3.2 Feature Encoding** 

- Categorical variables converted using one-hot encoding 

- Prevents models from interpreting categories as ordinal values 

## **3.3 Feature Scaling** 

StandardScaler applied 

Necessary for distance-based and gradient-based models 

## **4. Models Used** 

## **Regression Models** 

- Linear Regression 

- Polynomial Regression 

- K-Nearest Neighbors Regressor 

- Decision Tree Regressor 

## **Classi�cation Models** 

Logistic Regression 

- Support Vector Machine (SVM) 

- K-Nearest Neighbors Classi�er 

- Decision Tree Classi�er 

## **5. Model Evaluation Metrics** 

## **Regression Metrics** 

Mean Absolute Error (MAE) 

- Mean Squared Error (MSE) 

- R² Score 

## **Classi�cation Metrics** 

Accuracy 

Precision 

- Recall 

Confusion Matrix 

Training time and prediction (response) time were also recorded for each model. 

## **6. Results and Analysis** 

## **6.1 Regression Results** 

Polynomial Regression achieved the highest R² score but required signi�cantly more training time. KNN showed slow prediction time due to distance calculations at inference. 

Decision Trees provided strong performance with fast inference. 

## **6.2 Classi�cation Results** 

SVM achieved the highest accuracy but required more computation. 

Logistic Regression provided balanced performance and interpretability. 

Decision Trees were fast but more prone to over�tting. 

## **7. Visualization** 

Regression: Actual vs Predicted house prices 

- Classi�cation: Confusion matrix for Logistic Regression 

Each visualization matches the learning task and evaluation objective. 

## **8. Conclusion** 

The project demonstrates that: 

Preprocessing signi�cantly improves performance 

- Model choice impacts both accuracy and computational e�ciency 

- Scaling is essential for certain models 

- No single model is optimal for all tasks 

Model selection should balance performance, interpretability, and e�ciency depending on the application. 

