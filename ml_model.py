import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

class CarPricePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.label_encoders = {}
        
    def generate_sample_data(self, n_samples=1000):
        """Generate sample training data"""
        np.random.seed(42)
        
        brands = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Tata', 'Mahindra', 'Ford', 'Renault']
        models = ['Swift', 'i20', 'City', 'Innova', 'Nexon', 'XUV500', 'EcoSport', 'Duster']
        fuel_types = ['Petrol', 'Diesel', 'CNG']
        transmissions = ['Manual', 'Automatic']
        locations = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']
        
        data = {
            'brand': np.random.choice(brands, n_samples),
            'model': np.random.choice(models, n_samples),
            'year': np.random.randint(2010, 2024, n_samples),
            'fuel_type': np.random.choice(fuel_types, n_samples),
            'transmission': np.random.choice(transmissions, n_samples),
            'kms_driven': np.random.randint(5000, 150000, n_samples),
            'location': np.random.choice(locations, n_samples),
            'owner': np.random.randint(1, 4, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate car age
        df['car_age'] = 2024 - df['year']
        
        # Generate price based on features (realistic pricing logic)
        base_price = 500000  # Base price in INR
        
        # Brand multiplier
        brand_multiplier = {
            'Toyota': 1.3, 'Honda': 1.2, 'Hyundai': 1.1, 'Maruti': 1.0,
            'Tata': 0.9, 'Mahindra': 0.95, 'Ford': 1.05, 'Renault': 0.85
        }
        
        # Calculate price
        df['price'] = base_price
        df['price'] *= df['brand'].map(brand_multiplier)
        df['price'] *= (1 - df['car_age'] * 0.08)  # Depreciation
        df['price'] *= (1 - df['kms_driven'] / 1000000)  # Mileage impact
        df['price'] *= np.where(df['fuel_type'] == 'Diesel', 1.1, 1.0)  # Diesel premium
        df['price'] *= np.where(df['transmission'] == 'Automatic', 1.15, 1.0)  # Auto premium
        df['price'] += np.random.normal(0, 50000, n_samples)  # Random noise
        df['price'] = np.maximum(df['price'], 100000)  # Minimum price
        
        return df
    
    def preprocess_features(self, df):
        """Preprocess features for training"""
        # Create feature encoders
        categorical_features = ['brand', 'model', 'fuel_type', 'transmission', 'location']
        numerical_features = ['year', 'kms_driven', 'owner', 'car_age']
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', 'passthrough', categorical_features)
            ]
        )
        
        return preprocessor, categorical_features, numerical_features
    
    def train_model(self):
        """Train the car price prediction model"""
        print("Generating training data...")
        df = self.generate_sample_data(2000)
        
        # Prepare features and target
        feature_columns = ['brand', 'model', 'year', 'fuel_type', 'transmission', 'kms_driven', 'location', 'owner', 'car_age']
        X = df[feature_columns].copy()
        y = df['price']
        
        # Encode categorical variables
        categorical_columns = ['brand', 'model', 'fuel_type', 'transmission', 'location']
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Mean Absolute Error: ₹{mae:,.2f}")
        print(f"R² Score: {r2:.3f}")
        
        return mae, r2
    
    def predict_price(self, car_data):
        """Predict car price"""
        if not self.model:
            raise ValueError("Model not trained or loaded")
        
        # Prepare input data
        input_df = pd.DataFrame([car_data])
        input_df['car_age'] = 2024 - input_df['year']
        
        # Encode categorical variables
        categorical_columns = ['brand', 'model', 'fuel_type', 'transmission', 'location']
        
        for col in categorical_columns:
            if col in self.label_encoders:
                try:
                    input_df[col] = self.label_encoders[col].transform([str(car_data[col])])[0]
                except ValueError:
                    # Handle unseen categories
                    input_df[col] = 0
        
        # Make prediction
        feature_columns = ['brand', 'model', 'year', 'fuel_type', 'transmission', 'kms_driven', 'location', 'owner', 'car_age']
        X_input = input_df[feature_columns]
        
        predicted_price = self.model.predict(X_input)[0]
        
        # Calculate confidence score (simplified)
        confidence_score = min(0.95, max(0.65, 1.0 - (car_data['kms_driven'] / 200000) * 0.3))
        
        return predicted_price, confidence_score
    
    def save_model(self, model_path, encoders_path):
        """Save trained model and encoders"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoders, encoders_path)
        print(f"Model saved to {model_path}")
        print(f"Encoders saved to {encoders_path}")
    
    def load_model(self, model_path, encoders_path):
        """Load trained model and encoders"""
        if os.path.exists(model_path) and os.path.exists(encoders_path):
            self.model = joblib.load(model_path)
            self.label_encoders = joblib.load(encoders_path)
            print("Model and encoders loaded successfully")
            return True
        return False