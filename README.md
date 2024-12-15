## Diabetes Dataset Analysis

This repository contains an analysis of a diabetes dataset collected from the Iraqi population. The dataset was compiled using patient records from the Medical City Hospital laboratory and the Specialized Center for Endocrinology and Diabetes at Al-Kindy Teaching Hospital. The dataset provides valuable insights into diabetes classification and associated clinical parameters.

## Dataset Overview

The dataset contains the following attributes:

- **Patient Number**: Unique identifier for each patient.
- **Blood Sugar Levels**: Measured blood sugar levels.
- **Age**: Age of the patient.
- **Gender**: Male (M) or Female (F).
- **Creatinine Ratio (Cr)**: Laboratory measure of kidney function.
- **Body Mass Index (BMI)**: Indicator of body fat.
- **Urea**: Urea levels in the blood.
- **Cholesterol Levels**: Includes total cholesterol, LDL, HDL, and VLDL cholesterol levels.
- **Triglycerides (TG)**: Measure of fat in the blood.
- **HbA1c**: Long-term blood sugar measurement.
- **Diabetes Classification (CLASS)**: Categorical variable indicating if a patient is Diabetic (Y), Non-Diabetic (N), or Pre-Diabetic (P).

## Key Insights

### Data Cleanup

- **CLASS Attribute**: Issues with trailing spaces in class labels (e.g., `"Y "` instead of `"Y"`) were resolved.
- **Gender Attribute**: Ensured consistent formatting (e.g., converted `"f"` to `"F"`).

### Summary Statistics

- Dataset contains 1,000 entries.
- CLASS Distribution:
  - Diabetic (`Y`): 844
  - Non-Diabetic (`N`): 103
  - Pre-Diabetic (`P`): 53

- **Age Distribution**: Majority of patients are aged 50-59.
- **Gender Distribution**:
  - Male (`M`): 565
  - Female (`F`): 435

### Data Visualizations

1. **CLASS Distribution**: Bar chart visualizing the number of patients in each diabetes classification.
2. **Age Distribution by CLASS**: Box plots highlighting age differences across diabetes classes.
3. **Gender Distribution**: Bar chart of male and female counts.
4. **Heatmap**: Correlation matrix for numerical features, identifying relationships between variables.

### Feature Engineering

- **Age Range**: Age was grouped into bins (e.g., `20-29`, `30-39`, etc.) for better analysis.
- Irrelevant columns (`ID`, `No_Pation`, `AGE`) were removed for streamlined feature selection.

## Installation and Setup

### Required Libraries

Install the following Python libraries to run the analysis:

```bash
pip install pandas seaborn matplotlib shap scikit-learn xgboost
```

### Download Dataset

Download the dataset using the following command:

```bash
!wget https://data.mendeley.com/public-files/datasets/wj9rwkp9c2/files/2eb60cac-96b8-46ea-b971-6415e972afc9/file_downloaded
```

### Load the Dataset

```python
import pandas as pd

df = pd.read_csv('file_downloaded')
df.head()
```

## Analysis Highlights

### Feature Selection

Features selected for the analysis:
- **Numerical Features**: Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI.
- **Categorical Features**: Gender, Age Range.

Target variable: **CLASS**.

### Visualization Examples

1. **Age Distribution**:
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['AGE'], bins=10, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
```

2. **Correlation Heatmap**:
```python
import numpy as np
correlation_values = df.select_dtypes(include=np.number).corr()

plt.figure(figsize=(15, 9))
sns.heatmap(correlation_values, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

### Model Training

Several classification models were implemented to predict diabetes classes:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. Support Vector Machine (SVM)

### Performance Evaluation

Metrics used:
- Accuracy
- Confusion Matrix
- Classification Report

## Contributing

If you have suggestions for improving the analysis or additional features to include, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

For questions or feedback, please contact [Your Name/Email].


