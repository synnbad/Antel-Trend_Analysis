# Exploratory Data Analysis Workspace

This workspace is set up for comprehensive exploratory data analysis (EDA) using Python and Jupyter notebooks.

## 🚀 Quick Start

1. **Set up Python environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```

2. **Launch Jupyter Lab:**
   ```bash
   jupyter lab
   ```

3. **Start exploring your data!**

## 📁 Project Structure

```
├── data/
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned and transformed data
│   └── external/     # Third-party data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── 03_visualization.ipynb
├── src/
│   ├── data/         # Data loading and processing
│   ├── features/     # Feature engineering
│   └── visualization/ # Custom plotting functions
├── reports/
│   ├── figures/      # Generated plots and charts
│   └── summaries/    # Analysis reports
└── config/           # Configuration files
```

## 📊 Available Libraries

### Core Data Analysis
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing

### Visualization
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualization
- **plotly**: Interactive plots
- **altair**: Grammar of graphics
- **bokeh**: Interactive web-ready plots

### Machine Learning
- **scikit-learn**: Machine learning algorithms
- **statsmodels**: Statistical modeling
- **xgboost**: Gradient boosting
- **lightgbm**: Gradient boosting

### Jupyter Ecosystem
- **jupyterlab**: Modern notebook interface
- **ipywidgets**: Interactive widgets
- **notebook**: Classic Jupyter notebooks

## 🔧 Getting Started with EDA

1. **Load your data** into the `data/raw/` directory
2. **Create a new notebook** in the `notebooks/` directory
3. **Import common libraries:**
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   import plotly.express as px
   ```

## 📝 Best Practices

- Keep raw data immutable in `data/raw/`
- Document your analysis steps in notebooks
- Use descriptive variable names
- Save processed data in `data/processed/`
- Export visualizations to `reports/figures/`

## 🤝 Contributing

Feel free to add your own utility functions in the `src/` directory and share useful analysis patterns!
