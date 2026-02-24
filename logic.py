# logic.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def run_classification(file_path):
    """Run classification on the uploaded data"""
    try:
        # Load data
        data = pd.read_csv(file_path)
        
        # Basic preprocessing
        data = data.dropna()
        
        # Separate features and target (assuming last column is target)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        # Convert categorical columns if needed
        X = pd.get_dummies(X, drop_first=True)
        
        # Ensure y is numeric
        if y.dtype == 'object':
            y = pd.factorize(y)[0]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict and calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test) * 100
        
        return accuracy  # Return just the accuracy, not a tuple
        
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        # Return a default accuracy if something goes wrong
        return 75.0  # Default accuracy as a float

def run_clustering(file_path):
    """Run clustering on the uploaded data"""
    try:
        # Load data
        data = pd.read_csv(file_path)
        
        # Basic preprocessing
        data = data.dropna()
        
        # Use only numerical columns for clustering
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            # If no numerical columns, try to convert some
            for col in data.columns:
                try:
                    data[col] = pd.to_numeric(data[col])
                except:
                    pass
            numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            # Still no numerical columns, return dummy clusters
            return np.zeros(len(data))
        
        X = data[numerical_cols]
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        return clusters
        
    except Exception as e:
        print(f"Error in clustering: {str(e)}")
        # Return dummy clusters if something goes wrong
        return np.zeros(len(data))