# Credit Card Fraud Detection Dataset

## Overview
This repository contains a credit card transaction dataset commonly used for fraud detection analysis and machine learning projects. The dataset includes anonymized credit card transactions with features that can be used to identify fraudulent activities.

## Dataset Description

### Files
- `creditcard.csv` - Main dataset file containing transaction records
- `creditcard.csv.bz2` - Compressed version of the dataset

### Features
The dataset contains the following columns:
- **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset
- **V1-V28**: Anonymized features resulting from PCA transformation (to protect user identities and sensitive information)
- **Amount**: Transaction amount
- **Class**: Target variable (0 = legitimate transaction, 1 = fraudulent transaction)

### Dataset Characteristics
- Contains transactions made by European cardholders
- Highly imbalanced dataset (fraudulent transactions are much rarer than legitimate ones)
- All features are numerical
- PCA-transformed features maintain data privacy while preserving analytical value

## Usage

### Loading the Dataset

#### Python (Pandas)
```python
import pandas as pd

# Load the CSV file
df = pd.read_csv('creditcard.csv')

# Or load the compressed version
df = pd.read_csv('creditcard.csv.bz2', compression='bz2')

print(df.head())
print(df.info())
```

#### R
```r
# Load the CSV file
data <- read.csv('creditcard.csv')

# View the structure
str(data)
head(data)
```

## Potential Use Cases
- Binary classification (fraud vs. legitimate)
- Anomaly detection
- Imbalanced dataset handling techniques
- Machine learning model training and evaluation
- Feature engineering and selection
- Cost-sensitive learning

## Common Analysis Tasks
1. **Exploratory Data Analysis (EDA)**
   - Distribution of fraudulent vs. legitimate transactions
   - Transaction amount patterns
   - Time-based patterns

2. **Data Preprocessing**
   - Handling class imbalance (SMOTE, undersampling, oversampling)
   - Feature scaling
   - Train-test split with stratification

3. **Model Building**
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Neural Networks
   - Isolation Forest

4. **Evaluation Metrics**
   - Precision, Recall, F1-Score
   - ROC-AUC
   - Precision-Recall Curve
   - Confusion Matrix

## Important Considerations
- **Class Imbalance**: The dataset is highly imbalanced. Use appropriate techniques like SMOTE, class weights, or ensemble methods
- **Evaluation Metrics**: Accuracy is not a good metric for this dataset. Focus on Precision, Recall, F1-Score, and AUC-ROC
- **Privacy**: Features V1-V28 are already anonymized through PCA transformation

## Requirements
```
pandas
numpy
scikit-learn
matplotlib
seaborn
imbalanced-learn (for SMOTE)
```

## Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```



## Acknowledgments
This dataset is commonly used in the data science community for fraud detection research and education.

## Contributing
Feel free to fork this repository and submit pull requests for any improvements or additional analysis.

## Contact
For questions or suggestions, please open an issue in this repository.
