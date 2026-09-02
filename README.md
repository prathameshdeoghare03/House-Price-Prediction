# 🏠 House Price Prediction

A Machine Learning project that predicts **house sale prices** using property features from the **Ames Housing dataset**.

The project compares **Linear Regression** and **Random Forest Regression** models and evaluates their performance using **RMSE** and **R² score**.

---

## 📌 Project Overview

House prices depend on several factors such as living area, overall quality, garage capacity, basement features, and other property characteristics.

This project builds machine learning regression models to predict the `SalePrice` of houses based on these features.

The complete workflow includes:

**Data Loading → Data Exploration → Missing Value Handling → Encoding → Train-Test Split → Feature Scaling → Model Training → Evaluation → Feature Importance → Visualization → Cross-Validation**

---

## 🎯 Objectives

* Analyze the Ames Housing dataset.
* Explore house-price distributions and relationships.
* Handle missing values appropriately.
* Encode categorical variables.
* Split the dataset into training and testing data.
* Train regression models.
* Compare Linear Regression and Random Forest.
* Evaluate model performance.
* Identify important features affecting house prices.
* Visualize actual vs predicted prices.
* Perform 5-fold cross-validation.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data manipulation
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Scikit-learn** – Machine Learning

### Machine Learning Models

* Linear Regression
* Random Forest Regressor

---

## 📂 Project Structure

```text
House-Price-Prediction/
│
├── House-Price-Prediction.ipynb
├── AmesHousing.csv
└── README.md
```

---

## 📊 Dataset

The project uses the **Ames Housing dataset** stored as:

```text
AmesHousing.csv
```

The prediction target is:

```text
SalePrice
```

The dataset contains property-related features describing different aspects of residential properties.

Some important features explored in the notebook include:

* `Overall Qual`
* `Gr Liv Area`
* `Lot Frontage`
* `Garage Cars`
* `Garage Area`
* `Total Bsmt SF`
* `1st Flr SF`
* `2nd Flr SF`
* `Year Built`
* `Mas Vnr Area`
* Basement-related features
* Garage-related features

---

## 🔍 Exploratory Data Analysis

The notebook performs several exploratory analyses to understand the dataset and house prices.

### 📈 Distribution of House Prices

A histogram with KDE is used to visualize the distribution of `SalePrice`.

### 📦 SalePrice Boxplot

A boxplot is used to identify the spread and potential outliers in house prices.

### 🔥 Correlation Heatmap

A correlation heatmap is created using numerical features to analyze relationships between variables.

### 📊 Features Correlated with SalePrice

The notebook identifies the top features correlated with `SalePrice`.

### 🏡 Living Area vs Sale Price

A scatter plot examines the relationship between:

```text
Gr Liv Area → SalePrice
```

### ⭐ Overall Quality vs Sale Price

A boxplot examines how:

```text
Overall Qual → SalePrice
```

---

# 🧹 Data Preprocessing

## 1. Handling Missing Categorical Values

Missing values in features related to fireplaces, garages, and basements are replaced with meaningful categories.

Examples:

```text
No Fireplace
No Garage
No Basement
```

The notebook handles columns such as:

```text
Fireplace Qu
Garage Type
Garage Finish
Garage Qual
Garage Cond
Bsmt Qual
Bsmt Cond
Bsmt Exposure
BsmtFin Type 1
BsmtFin Type 2
```

---

## 2. Handling Missing Numerical Values

Several numerical features are filled with `0` where the absence of the feature represents zero.

Examples:

```text
Mas Vnr Area
Bsmt Full Bath
Bsmt Half Bath
BsmtFin SF 1
BsmtFin SF 2
Bsmt Unf SF
Total Bsmt SF
Garage Cars
Garage Area
Garage Yr Blt
```

For `Lot Frontage`, the notebook uses the median:

```python
df['Lot Frontage'] = df['Lot Frontage'].fillna(
    df['Lot Frontage'].median()
)
```

For `Electrical`, the mode is used:

```python
df['Electrical'] = df['Electrical'].fillna(
    df['Electrical'].mode()[0]
)
```

---

# 🔤 Categorical Encoding

Categorical variables are converted into numerical values using `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in X.select_dtypes(
    include=['object', 'string']
).columns:
    X[col] = le.fit_transform(X[col])
```

This allows categorical features to be used by the machine-learning models.

---

# ✂️ Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# 📏 Feature Scaling

`StandardScaler` is applied to the training and testing features.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then applied to the test data.

---

# 🤖 Machine Learning Models

## 1. Linear Regression

Linear Regression is used as the first baseline regression model.

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)
```

The model is evaluated using:

* RMSE
* R² Score

---

## 2. Random Forest Regressor

The second model is a Random Forest Regressor with **100 trees**.

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
```

Random Forest is also used to determine feature importance.

---

# 📏 Model Evaluation

The notebook evaluates regression models using two primary metrics.

## RMSE

**Root Mean Squared Error (RMSE)** measures the average magnitude of prediction errors.

```python
np.sqrt(
    mean_squared_error(y_test, predictions)
)
```

Lower RMSE indicates better prediction performance.

---

## R² Score

The R² score measures how well the model explains the variation in house prices.

```python
r2_score(y_test, predictions)
```

A higher R² score generally indicates better model performance.

---

# 🌲 Feature Importance

Random Forest feature importance is calculated using:

```python
importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
)
```

The notebook visualizes the **Top 10 most important features**:

```python
importance.sort_values(
    ascending=False
).head(10).plot(kind='barh')
```

This helps identify which property characteristics have the greatest influence on the model's predictions.

---

# 🔄 Cross-Validation

The Random Forest model is also evaluated using **5-fold cross-validation**.

```python
cv_rmse = np.sqrt(
    -cross_val_score(
        rf,
        X,
        y,
        scoring='neg_mean_squared_error',
        cv=5
    )
)

cv_rmse.mean()
```

The mean cross-validation RMSE provides an additional estimate of the model's performance across different subsets of the dataset.

---

# 📊 Actual vs Predicted Prices

The notebook compares actual house prices with Random Forest predictions using a scatter plot.

```text
Actual Prices
      │
      │       •
      │    •
      │  •
      │ •
      └──────────────── Predicted Prices
```

A reference diagonal line represents the ideal situation where:

```text
Actual Price = Predicted Price
```

The closer the predictions are to this line, the better the prediction performance.

---

# 🧠 Machine Learning Workflow

```text
                Ames Housing Dataset
                         │
                         ▼
                  Data Exploration
                         │
                         ▼
                Missing Value Handling
                         │
                         ▼
                  Feature Encoding
                         │
                         ▼
                   Train/Test Split
                         │
                         ▼
                  Feature Scaling
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Linear Regression      Random Forest
              │               Regressor
              │                     │
              └──────────┬──────────┘
                         ▼
                  Model Evaluation
                         │
                ┌────────┴────────┐
                ▼                 ▼
              RMSE              R² Score
                │                 │
                └────────┬────────┘
                         ▼
                Feature Importance
                         │
                         ▼
                  Cross Validation
                         │
                         ▼
              Actual vs Predicted
```

---

# 📌 Key Learning Outcomes

Through this project, I practiced:

* Data exploration
* Data cleaning
* Missing-value treatment
* Categorical encoding
* Feature scaling
* Regression modeling
* Linear Regression
* Random Forest Regression
* RMSE evaluation
* R² evaluation
* Feature importance analysis
* Correlation analysis
* Data visualization
* Cross-validation
* Actual vs predicted analysis

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/House-Price-Prediction.git
```

## 2. Navigate to the Project

```bash
cd House-Price-Prediction
```

## 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

## 5. Open the Notebook

Open:

```text
House-Price-Prediction.ipynb
```

Make sure `AmesHousing.csv` is available in the expected location and update the CSV path in the notebook if required.

---

# 🔮 Future Improvements

Possible improvements include:

* Use One-Hot Encoding instead of Label Encoding for nominal categorical variables.
* Apply advanced feature engineering.
* Perform hyperparameter tuning.
* Compare additional regression algorithms.
* Use Gradient Boosting or XGBoost.
* Apply cross-validation consistently during model selection.
* Analyze and handle outliers.
* Create a Streamlit interface for house-price prediction.
* Save and deploy the trained model.
* Add an interactive prediction dashboard.

---

# ⚠️ Disclaimer

This project is developed for **educational and machine-learning practice purposes**. The predictions are based on the available dataset and trained models and should not be considered professional real-estate valuation.

---

## 👨‍💻 Author

**Prathamesh Deoghare**

Computer Science & Engineering — Data Science

* GitHub: https://github.com/prathameshdeoghare03
* LinkedIn: `https://www.linkedin.com/in/prathamesh-deoghare-2524632b8/`

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.
