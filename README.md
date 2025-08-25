Trend analysis performed for the Antel project

##  Quick Start

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

##  Project Structure

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

