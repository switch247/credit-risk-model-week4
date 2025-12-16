Interim Project Report: Credit Risk Probability Model
Project: Credit Risk Probability Model for Alternative Data (Xente Transactions) Client: Bati Bank Date: December 2025
1. Understanding and Defining the Business Objective (Task 1)
1.1 Business Context and Problem Definition
The core objective is to build a credit scoring model for Bati Bank's new Buy Now, Pay Later (BNPL) service using alternative e-commerce transactional data (Xente dataset). The model must predict the Probability of Default (PD) to enable automated credit decisions and accurate risk-based pricing, ensuring sustainable product rollout.
1.2 Regulatory Alignment and Model Strategy
Compliance with the Basel Accords (II/III) is a primary driver for model design, as these frameworks demand high Interpretability and Governance.
Model Strategy Trade-offs: The initial strategy prioritizes an Interpretable Model (e.g., Logistic Regression with Weight of Evidence (WoE) transformation) as the baseline. This choice ensures stable, monotonic feature relationships and simplifies the documentation and audit trail required by regulators.
This interpretable approach is contrasted with Complex Models (e.g., Gradient Boosting and Random Forest). While complex models offer the potential for higher performance lift by capturing non-linear interactions, they introduce higher governance costs, require extensive use of explainability tooling (like SHAP), and carry an increased risk of overfitting to the noisy proxy target. Therefore, the strategy is to establish the simple, auditable Logistic Regression baseline first, then justify any transition to a complex model like Gradient Boosting with clear, evidence-based performance gains and robust governance controls.
1.3 Proxy Target Justification
Since the Xente data lacks an explicit 'default' label, a Proxy Target (is_high_risk) is being derived to enable supervised learning. This proxy uses RFM (Recency, Frequency, Monetary) analysis, labeling customers showing low engagement (low R, F, M) as high-risk/disengaged. The primary risk is that the model may predict customer disengagement rather than true financial default. Mitigation involves transparently documenting the proxy definition, using conservative thresholds for high-risk classification, and establishing continuous model monitoring.
2. Discussion of Completed Work and Initial Analysis (Tasks 2 & 3 Summary)
Tasks 1 (Business Understanding), 2 (EDA), and 3 (Feature Engineering Pipeline setup) are complete.
2.1 Exploratory Data Analysis (EDA) Insights
The initial analysis revealed critical data characteristics that inform the subsequent modeling approach.
Statistic
Value
Implication
Total Records
95,662
Robust dataset size.
Total Features
16
Manageable feature count.
Missing Values
0%
No imputation required for raw data.
Default Class
0.2%
Severe class imbalance requires specific handling.
Amount (Median)
1,000.00 UGX
Confirms heavy right-skewness.
Amount (Average)
6,717.85 UGX
Large difference from median indicates extreme outliers.

Key Data Insights and Implications (Revised for 5-Figure Support)
Missingness (Insight 1): Zero missing values, simplifying the data cleaning process.
Class Balance (Insight 2): Severe imbalance in the original target (0.2% fraud) necessitates targeted sampling or loss function modification in Task 5.
Numeric Skewness (Insight 3): The extreme right-skewness of Amount (visibly confirmed in Figure 4) mandates a Log Transformation to stabilize variance for the linear model.
Extreme Outliers (Insight 4): A large share of IQR outliers (visibly confirmed in Figure 5) necessitates an aggressive outlier treatment strategy (clipping) for stable model coefficients.
Feature Redundancy (Insight 5): A near-perfect linear relationship between Amount and Value (confirmed in Figure 3) requires feature selection to mitigate multicollinearity.
Categorical Cardinality (Insight 6): Categorical features exhibit concentration (Figure 1) or low effective cardinality (Figure 2), confirming suitability for WoE binning or One-Hot Encoding to maintain model interpretability.
2.2 Visualizations (Top 5 Key Figures)
To ensure professional documentation standards, all generated figures are stored as high-resolution assets (available in the EDA notebook). These five figures are explicitly selected to provide the visual evidence required to justify the core transformations and feature selections implemented in the Feature Engineering Pipeline (Task 3).
Figure 1: Top ProductCategory Values top_productcategory_values.png This bar chart supports Insight 6 (Categorical Cardinality) by demonstrating the high concentration and low effective cardinality of certain features (e.g., ProductCategory). The structure is ideal for applying Weight of Evidence (WoE) transformation to group sparse categories and establish monotonic, interpretable risk segments.
Figure 2: Top ChannelId Values top_channelid_values.png This visualization also supports Insight 6, showing high transaction volume concentration in primary channels for ChannelId. This low cardinality confirms the suitability for simple encoding strategies (like One-Hot Encoding), which preserves interpretability and avoids unnecessary complexity in the feature engineering process.
Figure 3: Pearson Correlation Heatmap pearson_correlation_heatmap.png The heatmap identifies a significant multicollinearity risk (Insight 5), showing a near-perfect linear relationship (~0.99) between Amount and Value. This finding justifies the feature selection decision to drop one of these redundant variables from the modeling set, thereby ensuring coefficient stability and model interpretability.
Figure 4: Amount Distribution (Linear Scale) amount_distribution_linear.png This histogram provides direct visual support for Insight 3 (Numeric Skewness). The extreme right-skewed distribution of the Amount feature explicitly mandates the Log Transformation within the pipeline to meet the linearity and variance assumptions of the baseline Logistic Regression model.
Figure 5: Amount Boxplot (Linear Scale) amount_boxplot_linear.png This boxplot visually confirms Insight 4 (Extreme Outliers), illustrating the large volume of values lying far beyond the Interquartile Range (IQR). This evidence justifies the necessity for an aggressive outlier treatment strategy (clipping/winsorization) to prevent these extreme values from destabilizing the model's coefficients.
2.3 Feature Engineering Pipeline Completion (Task 3)
The feature engineering pipeline, defined in xente_features.py, formalizes the transformation requirements identified in the EDA. Utilizing an sklearn ColumnTransformer structure for reproducibility, the pipeline ensures the data is correctly structured for the immediate next phase.
Temporal Extraction: Creation of features like hour, day, month, and year for the Recency component of RFM.
Customer Aggregates: Calculation of per-customer risk-related features (total amount, average amount, transaction count, and volatility) for the Monetary and Frequency components for the proxy target.
Preprocessing: Includes standard scaling for numerical features (in preparation for K-Means clustering in Task 4) and One-Hot Encoding for low-to-moderate cardinality categorical features.
3. Next Steps and Key Areas of Focus
Following the foundational data analysis and pipeline construction, the immediate future work focuses on the creation of the essential proxy target variable and the subsequent establishment of a regulated, robust baseline model.
Task ID
Description
Output/Deliverable
Focus Area
Task 4
Proxy Target Creation
Data merged with is_high_risk label.
RFM Analysis: Calculate RFM scores, use clustering (e.g., K-Means) to identify and assign the high-risk proxy label.
Task 5
Model Training & Comparison
Trained Baseline (LogReg + WoE) and Advanced (Random Forest) models.
Training: Apply WoE/IV transformation for baseline; use Resampling (SMOTE) for advanced model. Track all experiments and metrics using MLflow.
Task 6 (Initial)
Deployment Preparation
Define FastAPI endpoint and model artifact serialization.
Define a FastAPI endpoint for inference, serialize the model and pipeline via MLflow, and package the entire service into a Docker container. Finally, configure the CI/CD pipeline (using GitHub Actions) for automated testing and deployment initiation.

Key Risk: The project's overall predictive performance remains highly dependent on the quality and robustness of the RFM-derived proxy target (Task 4).

