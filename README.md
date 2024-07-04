# Big Mart Sales Prediction

## Project Overview

This project aims to predict the sales for various products at Big Mart outlets using machine learning techniques. It encompasses data preprocessing, exploratory data analysis (EDA), feature engineering, and model building.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Data Preprocessing](#data-preprocessing)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Feature Engineering](#feature-engineering)
5. [Model Building](#model-building)
6. [Results](#results)
7. [Dependencies](#dependencies)
8. [Usage](#usage)
9. [License](#license)

## Project Structure

```
BigMartSalesPrediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Preprocessing.ipynb
│   ├── FeatureEngineering.ipynb
│   └── Modeling.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   └── modeling.py
│
└── README.md
```

## Data Preprocessing

Data preprocessing was crucial in preparing the dataset for analysis and model building. The following steps were taken:

- **Data Cleaning:** Removed irrelevant columns: `Item_Identifier` and `Outlet_Identifier`.
- **Handling Missing Values:** Imputed missing values in the `Item_Weight` column with the mean value to maintain data integrity.

**Code:**

```python
# Removing irrelevant columns
data = data.drop(["Item_Identifier", "Outlet_Identifier"], axis=1)

# Filling missing values in Item_Weight
data["Item_Weight"].fillna(data["Item_Weight"].mean(), inplace=True)
```

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) was conducted to understand the dataset better and identify patterns or anomalies:

- **Descriptive Statistics:** Analyzed data distribution.
- **Visualizations:**
  - **Histograms:** To explore the distribution of numerical features.
  - **Boxplots:** To detect and analyze outliers.
  - **Distplot:** To examine the distribution of `Item_Weight`.
  - **Count Plot:** To visualize the frequency of categorical variables.

**Visualizations:**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram for all columns
data.hist(bins=30, figsize=(20, 15))
plt.show()

# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=data)
plt.show()

# Distplot for Item_Weight
sns.distplot(data["Item_Weight"], kde=True, hist=True)
plt.show()

# Count plot for categorical variables
sns.countplot(x="Item_Fat_Content", data=data)
plt.show()
```

## Feature Engineering

Transformed categorical variables into numerical format using label encoding to prepare them for model training:

- **Label Encoding:** Applied to `Item_Fat_Content`, `Item_Type`, `Outlet_Size`, `Outlet_Location_Type`, and `Outlet_Type`.

**Code:**

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for column in ["Item_Fat_Content", "Item_Type", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"]:
    data[column] = le.fit_transform(data[column])
```

## Model Building

Built and evaluated predictive models using machine learning algorithms:

1. **Random Forest:** Utilized to leverage its ensemble learning for accurate predictions.
2. **AdaBoost:** Applied to improve model performance by combining multiple weak learners.

**Code:**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error

# Splitting the data
X = data.drop("Item_Outlet_Sales", axis=1)
y = data["Item_Outlet_Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("Random Forest MSE:", mean_squared_error(y_test, rf_preds))

# AdaBoost
ada = AdaBoostRegressor()
ada.fit(X_train, y_train)
ada_preds = ada.predict(X_test)
print("AdaBoost MSE:", mean_squared_error(y_test, ada_preds))
```

## Results

- **Random Forest:** Provided robust predictions with good accuracy.
- **AdaBoost:** Enhanced prediction performance by combining multiple weak learners.

**Metrics:**

- Mean Squared Error (MSE) for Random Forest and AdaBoost was calculated to evaluate model performance.

## Dependencies

The project relies on the following libraries:

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

**Installation:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/Osama-Abo-Bakr/BigMartSalesPrediction.git
    ```

2. Navigate to the project directory:

    ```bash
    cd BigMartSalesPrediction
    ```

3. Explore the Jupyter notebooks in the `notebooks/` directory to understand the data preprocessing, EDA, feature engineering, and model building.

4. Run the scripts in the `src/` directory to preprocess data and build models.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to the project by opening issues or submitting pull requests!

---

This `README.md` file provides a comprehensive overview of the project, detailing the steps and code used for each phase, and makes it easy for others to understand and replicate your work.
