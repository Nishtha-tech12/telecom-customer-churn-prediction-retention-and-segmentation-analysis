# Telecom Customer Churn Prediction, Retention and Segmentation Analysis

An end-to-end Machine Learning project structured into separate, modular workflows. This project analyzes subscriber demographics and contract features to predict customer churn, optimizes a tree-based ensemble architecture to maximize predictive sensitivity, and groups customers into business personas using unsupervised K-Means clustering.

---

##  Project Structure & Step-by-Step Workflow

### 1️) notebooks/01_eda.ipynb (Exploratory Data Analysis)
* **What Was Done**: Explored the raw characteristics of the dataset (7043 rows and 33 columns). Visualized individual continuous variables and plotted categorical variables against customer turnover using seaborn and matplotlib. Computed descriptive statistics and initial correlation arrays.
* **Key Visualizations & Text Insights Added**:
  * **Tenure Months Distribution**: Highlighted high density at both ends (under 5 months and near 72 months). *Insight: The first year is the high-risk danger zone for customer turnover.*
  * **Contract Type vs Churn Count**: Proved that month-to-month contracts account for almost all churn (~42.7%), while two-year contracts are extremely stable (under 3% churn).
  * **Tech Support Impact**: Found that subscribers without Tech Support have an aggressively higher likelihood of leaving the company.
  * **Price Sensitivity**: Boxplots revealed that churned customers have a higher median monthly cost (around $80) than retained customers (around $65).

### 2️) notebooks/02_data_cleaning.ipynb (Data Wrangling)
* **What Was Done**: Addressed a critical data type anomaly where Total Charges was read as text because of empty string characters left by new clients with zero tenure. Dropped metadata fields and variables that cause target data leakage. Converted categoricals into numerical values.
* **Key Outputs**:
  * Converted Total Charges to a numeric float type and filled all missing values with 0.
  * Dropped tracking identifiers (CustomerID, Lat Long, Zip Code, City) along with target-leaking features (Churn Score, CLTV, Churn Reason).
  * Processed text variables using pd.get_dummies to create the final data frame.
  * Saved and exported the matrices as X_clean.csv and Y_clean.csv for the modeling pipeline.

### 3️) notebooks/03_machine_learning_implementation.ipynb (Supervised Learning)
* **What Was Done**: Partitioned data into an 80/20 train-test split. Tested a baseline Random Forest model, addressed severe class imbalance using class weighting, calculated feature importances, and ran an extensive manual grid sweep across different numbers of trees and tree depths to explicitly maximize Recall. Verified stability using 5-Fold Cross-Validation and evaluated model boundaries with an ROC curve.
* **Key Outputs & Evaluation Achievements**:
  * **Baseline Model**: Achieved high accuracy but poor recall, completely missing the minority churn class due to imbalanced data (5,174 No vs 1,869 Yes).
  * **Approach 1 (Class Imbalance Fix)**: Applied class_weight='balanced', which successfully forced the trees to prioritize catching the minority churn cases.
  * **Approach 2 & 2 Again (Hyperparameter Sweeping)**: Tested combinations across n_estimators_list and max_depth_list. The optimized balanced configuration (n_estimators=300, max_depth=10) achieved a strong recall of ~80% on the churn class and an overall test accuracy of ~76%.
  * **Approach 3 (Feature Selection)**: Ranked column importance, identifying Tenure Months, Total Charges, Monthly Charges, and Contract_Two year as the top operational drivers of subscriber turnover.
  * **Cross-Validation**: Confirmed model stability across different subsets of the data with a Mean CV Accuracy of ~76% and a Mean CV Recall of ~80%.
  * **ROC-AUC Boundary**: Achieved a solid Area Under the Curve score of ~0.84, validating strong model classification power.

### 4️) notebooks/04_customer_segmentation.ipynb (Unsupervised Clustering)
* **What Was Done**: Extracted raw continuous prediction probabilities from the tuned Random Forest model. Standardized the metrics using StandardScaler and ran a K-Means clustering algorithm loop from K=1 to 15. Used an Elbow Plot (Within-Cluster Sum of Squares) to select the perfect number of clusters to define customer risk profiles.
* **Key Outputs & Segment Personas**:
  * **WCSS Elbow Curve Plot**: The curve showed a distinct bend ("elbow") right at K=3, indicating the optimal mathematical grouping.
  * Generated 3 distinct subscriber risk segments by mapping profiles against model probabilities:
    1. **Cluster 0: Budget Loyal Customers**: High tenure, low monthly costs, and extremely low churn probability.
    2. **Cluster 1: High Risk New Customers**: Short tenure, high monthly fees, and the highest churn probability.
    3. **Cluster 2: Loyal Premium Customers**: High tenure, high monthly investment, and strong loyalty.

---

##  Strategic Business Takeaways

1. **Front-Load Retention Actions**: Since churn risk is heavily concentrated within the first 12 months, customer onboarding teams must focus their efforts on new users during their first 90 days.
2. **Eliminate Billing Friction**: Month-to-month accounts and manual payment methods (like electronic checks) cause customers to re-evaluate their expenses every single month. Marketing teams should offer financial incentives or discounts to migrate users to 1-year or 2-year automated payment contracts.
3. **Isolate High-Risk Personas**: Using the K-Means groupings, customer support teams can automatically isolate the "High Risk New Customers" cluster and proactively offer targeted support trials or custom service packages to prevent early subscriber attrition.
