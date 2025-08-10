# Student Performance Prediction Model

A machine learning project that predicts student math scores based on various features like gender, ethnicity, parental education, lunch type, and test preparation.

## 🚀 Quick Start

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run streamlit_app.py
```

## 🔧 Troubleshooting

### Scikit-learn Version Compatibility Issues

If you encounter errors like:
- `No module named 'sklearn.ensemble._gb_losses'`
- `local variable 'e' referenced before assignment`
- `'NoneType' object has no attribute 'tb_frame'`

**Solution**: Run the compatibility fixer:
```bash
python fix_compatibility.py
```

This script will:
- Check your current scikit-learn version
- Install the compatible version (1.3.2)
- Test if the model loads correctly

### Manual Fix

If the automatic fix doesn't work, manually install the compatible version:
```bash
pip install scikit-learn==1.3.2
```

## 📁 Project Structure

```
Project 1/
├── artifacts/           # Saved models and data
│   ├── model.pkl       # Trained model
│   ├── preprocessor.pkl # Data preprocessor
│   └── *.csv          # Dataset files
├── src/                # Source code
│   ├── components/     # ML pipeline components
│   ├── pipeline/       # Training and prediction pipelines
│   ├── utils.py        # Utility functions
│   └── exception.py    # Custom exception handling
├── notebooks/          # Jupyter notebooks for EDA and training
├── streamlit_app.py    # Web application
├── retrain_model.py    # Script to retrain the model
├── fix_compatibility.py # Version compatibility fixer
└── requirements.txt    # Dependencies
```

## 🎯 Features

- **Gender**: Male/Female
- **Race/Ethnicity**: Group A, B, C, D, E
- **Parental Education**: Various education levels
- **Lunch Type**: Standard/Free-reduced
- **Test Preparation**: Completed/None
- **Reading Score**: 0-100
- **Writing Score**: 0-100

## 📊 Model Performance

The model uses ensemble methods and achieves good prediction accuracy for student math scores.

## 🔄 Retraining

To retrain the model with current dependencies:
```bash
python retrain_model.py
```

## 📝 Notes

- The model was trained with scikit-learn 1.3.2
- Use the compatibility fixer if you encounter version issues
- The application uses Streamlit for the web interface
