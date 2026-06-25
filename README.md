\# Telecom Customer Churn Prediction, Retention and Segmentation Analysis



An end-to-end Machine Learning project structured into separate, modular workflows. This project analyzes subscriber demographics and contract features to predict customer churn, optimizes a tree-based ensemble architecture to maximize predictive sensitivity, and groups customers into business personas using unsupervised K-Means clustering.



\---



\##  Project Structure \& Step-by-Step Workflow



\### 1️⃣ notebooks/01\_eda.ipynb (Exploratory Data Analysis)

\* \*\*What Was Done\*\*: Explored the raw characteristics of the dataset (7043 rows and 33 columns). Visualized individual continuous variables and plotted categorical variables against customer turnover using seaborn and matplotlib. Computed descriptive statistics and initial correlation arrays.

\* \*\*Key Visualizations \& Text Insights Added\*\*:

&#x20; \* \*\*Tenure Months Distribution\*\*: Highlighted high density at both ends (under 5 months and near 72 months). \*Insight: The first year is the high-risk danger zone for customer turnover.

&#x20; \* \*\*Contract Type vs Churn Count\*\*: Proved that month-to-month contracts account for almost all churn (\~42.7%), while two-year contracts are extremely stable (under 3% churn).

&#x20; \* \*\*Tech Support Impact\*\*: Found that subscribers without Tech Support have an aggressively higher likelihood of leaving the company.

&#x20; \* \*\*Price Sensitivity\*\*: Boxplots revealed that churned customers have a higher median monthly cost (\~$80) than retained customers (\~$65).



\### 2️⃣ notebooks/02\_data\_cleaning.ipynb (Data Wrangling)

\* \*\*What Was Done\*\*: Addressed a critical data type anomaly where Total Charges was read as text because of empty string characters left by new clients with zero tenure. Dropped metadata fields and variables that cause target data leakage. Converted categoricals into numerical values.

\* \*\*Key Outputs\*\*:

&#x20; \* Converted Total Charges to a numeric float type and filled all missing values with 0.

&#x20; \* Dropped tracking identifiers (CustomerID, Lat Long, Zip Code, City) along with target-leaking features (Churn Score, CLTV, Churn Reason).

&#x20; \* Processed text variables using pd.get\_dummies to create the final data frame.

&#x20; \* Saved and exported the matrices as X\_clean.csv and Y\_clean.csv for the modeling pipeline.



\### 3️⃣ notebooks/03\_machine\_learning\_implementation.ipynb (Supervised Learning)

\* \*\*What Was Done\*\*: Partitioned data into an 80/20 train-test split. Tested a baseline Random Forest model, addressed severe class imbalance using class weighting, calculated feature importances, and ran an extensive manual grid sweep across different numbers of trees and tree depths to explicitly maximize Recall. Verified stability using 5-Fold Cross-Validation and evaluated model boundaries with an ROC curve.

\* \*\*Key Outputs \& Evaluation Achievements\*\*:

&#x20; \* \*\*Baseline Model\*\*: Achieved high accuracy but poor recall, completely missing the minority churn class due to imbalanced data (5,174 No vs 1,869 Yes).

&#x20; \* \*\*Approach 1 (Class Imbalance Fix)\*\*: Applied class\_weight='balanced', which successfully forced the trees to prioritize catching the minority churn cases.

&#x20; \* \*\*Approach 2 \& 2 Again (Hyperparameter Sweeping)\*\*: Tested combinations across n\_estimators\_list and max\_depth\_list. The optimized balanced configuration (n\_estimators=300, max\_depth=10) achieved a strong recall of \~80% on the churn class and an overall test accuracy of \~76%.

&#x20; \* \*\*Approach 3 (Feature Selection)\*\*: Ranked column importance, identifying Tenure Months, Total Charges, Monthly Charges, and Contract\_Two year as the top operational drivers of subscriber turnover.

&#x20; \* \*\*Cross-Validation\*\*: Confirmed model stability across different subsets of the data with a Mean CV Accuracy of \~76% and a Mean CV Recall of \~80%.

&#x20; \* \*\*ROC-AUC Boundary\*\*: Achieved a solid Area Under the Curve score of \~0.84, validating strong model classification power.



\### 4️⃣ notebooks/04\_customer\_segmentation.ipynb (Unsupervised Clustering)

\* \*\*What Was Done\*\*: Extracted raw continuous prediction probabilities from the tuned Random Forest model. Standardized the metrics using StandardScaler and ran a K-Means clustering algorithm loop from K=1 to 15. Used an Elbow Plot (Within-Cluster Sum of Squares) to select the perfect number of clusters to define customer risk profiles.

\* \*\*Key Outputs \& Segment Personas\*\*:

&#x20; \* \*\*WCSS Elbow Curve Plot\*\*: The curve showed a distinct bend ("elbow") right at K=3, indicating the optimal mathematical grouping.

&#x20; \* Generated 3 distinct subscriber risk segments by mapping profiles against model probabilities:

&#x20;   1. \*\*Cluster 0: Budget Loyal Customers\*\*: High tenure, low monthly costs, and extremely low churn probability.

&#x20;   2. \*\*Cluster 1: High Risk New Customers\*\*: Short tenure, high monthly fees, and the highest churn probability.

&#x20;   3. \*\*Cluster 2: Loyal Premium Customers\*\*: High tenure, high monthly investment, and strong loyalty.



\---



\##  Strategic Business Takeaways



1\. \*\*Front-Load Retention Actions\*\*: Since churn risk is heavily concentrated within the first 12 months, customer onboarding teams must focus their efforts on new users during their first 90 days.

2\. \*\*Eliminate Billing Friction\*\*: Month-to-month accounts and manual payment methods (like electronic checks) cause customers to re-evaluate their expenses every single month. Marketing teams should offer financial incentives or discounts to migrate users to 1-year or 2-year automated payment contracts.

3\. \*\*Isolate High-Risk Personas\*\*: Using the K-Means groupings, customer support teams can automatically isolate the "High Risk New Customers" cluster and proactively offer targeted support trials or custom service packages to prevent early subscriber attrition.



