🏠 House Price Prediction Project


📌 Overview
Build a machine learning model in Python to predict house sale prices from various property features. Learn the full data science process: data loading, cleaning, EDA, feature encoding, model training & evaluation.

📂 Repository Structure

Edit
├── data/


│   └── HousePriceDataset.xlsx


├── notebooks/
│   └── exploration.ipynb
├── src/
│   └── preprocess.py
│   └── train_model.py
├── README.md
└── requirements.txt
🚀 Setup & Installation

git clone <your-repo-url>
cd house-price-prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
📊 Dataset Description
The dataset includes 13 columns such as:

LotArea, MSZoning, OverallCond, YearBuilt, TotalBsmtSF, SalePrice, etc.


📌 Workflow Steps
1. Importing Libraries & Loading Data
Load essential packages: pandas, numpy, matplotlib, seaborn.

2. Data Cleaning & Preprocessing
Drop irrelevant fields (e.g., Id)

Handle missing values by imputation or row removal

Separate categorical vs numerical features for transformation
GeeksforGeeks

3. Exploratory Data Analysis (EDA)
Visualize correlations (heatmaps), distributions, and frequency counts

Use barplots for categorical variable distribution
GeeksforGeeks

4. Feature Encoding
Convert categorical columns to numeric via OneHotEncoder, merging back to complete dataframe
GeeksforGeeks

5. Train/Test Split
Split dataset into training (80%) and validation (20%) sets
GeeksforGeeks

6. Model Building & Evaluation
Compare regression models:

Support Vector Regression (SVR)

Random Forest Regressor

Linear Regression

Evaluate based on Mean Absolute Percentage Error (MAPE)
SVR achieved the lowest error (~ 0.187), slightly outperforming linear regression; Random Forest was ~ 0.193 
arxiv.org


🧑‍💻 How to Run
Preprocessing: python src/preprocess.py

Train Models: python src/train_model.py

Expected output: trained models, validation metrics (MAPE) printed for each algorithm.

📈 Results
SVR: MAPE ≈ 0.187

Linear Regression: MAPE ≈ 0.1874

Random Forest: MAPE ≈ 0.1929
SVR is the most accurate in this case, but ensemble methods like boosting (e.g., XGBoost) may further improve performance 
GeeksforGeeks
arxiv.org

🔄 Extensions & Improvements
Hyperparameter tuning (e.g., grid search)

Try XGBoost, Gradient Boosting, Bagging

Incorporate ensemble stacking

Explore feature engineering: interaction terms, polynomial features

Deploy model using Flask, Streamlit, or FastAPI

