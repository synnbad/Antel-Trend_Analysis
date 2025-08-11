"""
Data loading utilities for the EDA workspace.

This module provides functions for loading various data formats
and performing initial data validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Union, Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_csv_data(
    file_path: Union[str, Path], 
    **kwargs
) -> pd.DataFrame:
    """
    Load CSV data with error handling and logging.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv()
    
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        logger.info(f"Loading CSV data from {file_path}")
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        raise

def load_excel_data(
    file_path: Union[str, Path], 
    sheet_name: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load Excel data with error handling and logging.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Name of the sheet to load (default: first sheet)
        **kwargs: Additional arguments to pass to pd.read_excel()
    
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        logger.info(f"Loading Excel data from {file_path}")
        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading Excel data: {e}")
        raise

def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about a DataFrame.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dict with data information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    return info

def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform basic data validation checks.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dict with validation results
    """
    validation_results = {
        'is_empty': len(df) == 0,
        'has_missing_values': df.isnull().any().any(),
        'has_duplicates': df.duplicated().any(),
        'column_count': len(df.columns),
        'row_count': len(df),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    
    return validation_results
