# Credit Card Fraud Detection Using Machine Learning

## Overview
This project implements a credit card fraud detection system using machine learning algorithms. The project compares three different classification models (Logistic Regression, Decision Tree, and Random Forest) and explores both undersampling and oversampling (SMOTE) techniques to handle imbalanced data.

## Project Workflow

### 1. Data Loading and Exploration
- Load the credit card transaction dataset
- Display dataset shape, info, and statistics
- Check for missing values and duplicates

### 2. Data Preprocessing
- **Feature Scaling**: Standardized the 'Amount' column using StandardScaler
- **Feature Selection**: Dropped the 'Time' column
- **Duplicate Removal**: Removed duplicate transactions

### 3. Handling Imbalanced Data
The dataset is highly imbalanced with far more legitimate transactions than fraudulent ones. Two approaches were implemented:

#### Undersampling
- Randomly sampled 473 normal transactions to match the number of fraudulent transactions
- Created a balanced dataset with equal representation of both classes

#### Oversampling (SMOTE)
- Used Synthetic Minority Over-sampling Technique (SMOTE)
- Generated synthetic samples for the minority class (fraudulent transactions)
- Balanced the dataset without losing information from the majority class

### 4. Model Training and Evaluation
Three machine learning models were trained and compared:

#### Models Used:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**

#### Evaluation Metrics:
- Accuracy Score
- Precision Score
- Recall Score
- F1 Score

### 5. Model Comparison
- Created comparison visualizations using bar plots
- Compared model performance across different sampling techniques
- Selected the best performing model

### 6. Model Deployment
- Trained final Random Forest model on the complete oversampled dataset
- Saved the model using joblib for future predictions
- Implemented prediction function to classify new transactions

## Dataset Description

### Features
- **Time**: Number of seconds elapsed between transactions
- **V1-V28**: Anonymized features from PCA transformation
- **Amount**: Transaction amount (standardized)
- **Class**: Target variable (0 = Normal, 1 = Fraudulent)

## Installation and Requirements

### Required Libraries
```bash
pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn joblib
```

### Dependencies
```
pandas
numpy
scikit-learn
seaborn
matplotlib
imbalanced-learn
joblib
```

## Usage

### Running the Notebook
1. Open `credit_card (1).ipynb` in Jupyter Notebook or JupyterLab
2. Update the file path to your dataset location
3. Run all cells sequentially

### Making Predictions
```python
import joblib

# Load the saved model
model = joblib.load("credid_card_model")

# Make prediction (provide 29 features)
pred = model.predict([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])

if pred == 0:
    print("Normal transaction")
else:
    print("Fraudulent transaction")
```

## Results

The project successfully demonstrates:
- Effective handling of imbalanced datasets using both undersampling and oversampling
- Comparison of multiple machine learning algorithms
- Model evaluation using multiple metrics
- Deployment-ready fraud detection model

## Files in Repository
- `creditcard.csv` - Main dataset
- `creditcard.csv.bz2` - Compressed dataset
- `credit_card (1).ipynb` - Jupyter notebook with complete implementation
- `credid_card_model` - Saved Random Forest model
- `README.md` - Project documentation

## Key Insights
- SMOTE oversampling generally provides better results than undersampling as it doesn't discard data
- Random Forest typically performs well for fraud detection tasks
- Multiple evaluation metrics are crucial for imbalanced classification problems
- Feature scaling improves model performance

## Future Improvements
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Try additional algorithms (XGBoost, Neural Networks)
- Implement cross-validation for more robust evaluation
- Feature importance analysis
- Real-time fraud detection pipeline

## Contributing
Feel free to fork this repository and submit pull requests for improvements.

## Contact
For questions or suggestions, please open an issue in this repository.
