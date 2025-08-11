"""
Visualization utilities for exploratory data analysis.

This module provides functions for creating common EDA visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Union
import warnings

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_missing_values(
    df: pd.DataFrame, 
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot missing values heatmap and bar chart.
    
    Args:
        df: Input DataFrame
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Missing values heatmap
    sns.heatmap(df.isnull(), cbar=True, ax=axes[0], cmap='viridis')
    axes[0].set_title('Missing Values Heatmap')
    
    # Missing values bar chart
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    
    if len(missing_counts) > 0:
        missing_counts.plot(kind='bar', ax=axes[1])
        axes[1].set_title('Missing Values Count by Column')
        axes[1].tick_params(axis='x', rotation=45)
    else:
        axes[1].text(0.5, 0.5, 'No missing values', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Missing Values Count by Column')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(
    df: pd.DataFrame, 
    figsize: Tuple[int, int] = (10, 8),
    method: str = 'pearson'
) -> None:
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: Input DataFrame
        figsize: Figure size
        method: Correlation method ('pearson', 'spearman', 'kendall')
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("No numeric columns found for correlation analysis.")
        return
    
    plt.figure(figsize=figsize)
    correlation_matrix = numeric_df.corr(method=method)
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f')
    plt.title(f'{method.capitalize()} Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_distributions(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> None:
    """
    Plot distributions for numeric columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to plot (default: all numeric columns)
        figsize: Figure size
    """
    # Select numeric columns
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [col for col in columns if col in df.columns and 
                       df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    if not numeric_cols:
        print("No numeric columns found for distribution plots.")
        return
    
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 4)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, col in enumerate(numeric_cols):
        row = i // n_cols
        col_idx = i % n_cols
        
        ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
        
        # Plot histogram with KDE
        df[col].hist(bins=30, alpha=0.7, ax=ax, density=True)
        df[col].plot(kind='kde', ax=ax, secondary_y=False)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
    
    # Hide empty subplots
    for i in range(len(numeric_cols), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None,
    max_categories: int = 20,
    figsize: Optional[Tuple[int, int]] = None
) -> None:
    """
    Plot distributions for categorical columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to plot (default: all categorical columns)
        max_categories: Maximum number of categories to display
        figsize: Figure size
    """
    # Select categorical columns
    if columns is None:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        cat_cols = [col for col in columns if col in df.columns and 
                   df[col].dtype in ['object', 'category']]
    
    if not cat_cols:
        print("No categorical columns found.")
        return
    
    n_cols = min(2, len(cat_cols))
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (n_cols * 8, n_rows * 6)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, col in enumerate(cat_cols):
        row = i // n_cols
        col_idx = i % n_cols
        
        ax = axes[row][col_idx] if n_rows > 1 else axes[col_idx]
        
        # Get value counts
        value_counts = df[col].value_counts().head(max_categories)
        
        # Plot bar chart
        value_counts.plot(kind='bar', ax=ax)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for i in range(len(cat_cols), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        ax = axes[row][col_idx] if n_rows > 1 else axes[col_idx]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.show()

def create_eda_summary_plot(df: pd.DataFrame) -> None:
    """
    Create a comprehensive EDA summary plot.
    
    Args:
        df: Input DataFrame
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Dataset overview
    ax1 = plt.subplot(2, 3, 1)
    info_text = f"""Dataset Overview:
    Shape: {df.shape}
    Columns: {len(df.columns)}
    Rows: {len(df)}
    Missing Values: {df.isnull().sum().sum()}
    Duplicates: {df.duplicated().sum()}
    Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"""
    
    ax1.text(0.1, 0.5, info_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='center')
    ax1.set_title('Dataset Overview')
    ax1.axis('off')
    
    # Data types
    ax2 = plt.subplot(2, 3, 2)
    dtype_counts = df.dtypes.value_counts()
    dtype_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
    ax2.set_title('Data Types Distribution')
    
    # Missing values
    ax3 = plt.subplot(2, 3, 3)
    missing_percent = (df.isnull().sum() / len(df) * 100)
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
    
    if len(missing_percent) > 0:
        missing_percent.head(10).plot(kind='bar', ax=ax3)
        ax3.set_title('Missing Values % (Top 10)')
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'No missing values', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Missing Values %')
    
    # Numeric columns distribution
    ax4 = plt.subplot(2, 3, 4)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Show distribution of the first numeric column
        df[numeric_cols[0]].hist(bins=30, ax=ax4, alpha=0.7)
        ax4.set_title(f'Distribution: {numeric_cols[0]}')
    else:
        ax4.text(0.5, 0.5, 'No numeric columns', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Numeric Distribution')
    
    # Categorical columns
    ax5 = plt.subplot(2, 3, 5)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        # Show distribution of the first categorical column
        value_counts = df[cat_cols[0]].value_counts().head(10)
        value_counts.plot(kind='bar', ax=ax5)
        ax5.set_title(f'Top 10: {cat_cols[0]}')
        ax5.tick_params(axis='x', rotation=45)
    else:
        ax5.text(0.5, 0.5, 'No categorical columns', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Categorical Distribution')
    
    # Correlation heatmap (for numeric columns only)
    ax6 = plt.subplot(2, 3, 6)
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, ax=ax6, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        ax6.set_title('Correlation Matrix')
    else:
        ax6.text(0.5, 0.5, 'Insufficient numeric columns\nfor correlation', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
