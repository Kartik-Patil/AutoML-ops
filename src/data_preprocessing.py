import pandas as pd
import numpy as np

def preprocess_data(df):
    try:
        # Make a copy of the dataframe
        df = df.copy()
        
        # Identify numeric and non-numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
        
        # Handle numeric columns
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, invalid values become NaN
            df[col] = df[col].fillna(df[col].mean())
            
        # Handle non-numeric columns
        for col in non_numeric_columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'missing')
        
        # Convert categorical variables to numerical
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            df[column] = pd.Categorical(df[column]).codes
            
        return df
        
    except Exception as e:
        raise Exception(f"Error in preprocessing: {str(e)}")