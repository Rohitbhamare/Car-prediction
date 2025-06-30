from pymongo import MongoClient
from config import Config
import datetime

class Database:
    def __init__(self):
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client[Config.DATABASE_NAME]
        self.predictions = self.db.predictions
        self.cars_data = self.db.cars_data
        
    def save_prediction(self, car_data, predicted_price, confidence_score):
        """Save prediction to database"""
        prediction_doc = {
            'car_data': car_data,
            'predicted_price': predicted_price,
            'confidence_score': confidence_score,
            'timestamp': datetime.datetime.utcnow()
        }
        return self.predictions.insert_one(prediction_doc)
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions"""
        return list(self.predictions.find().sort('timestamp', -1).limit(limit))
    
    def get_predictions_by_brand(self, brand):
        """Get predictions filtered by brand"""
        return list(self.predictions.find({'car_data.brand': brand}))
    
    def save_car_data(self, car_data):
        """Save car data for training"""
        car_doc = {
            **car_data,
            'timestamp': datetime.datetime.utcnow()
        }
        return self.cars_data.insert_one(car_doc)
    
    def get_cars_data(self, limit=None):
        """Get cars data for analysis"""
        if limit:
            return list(self.cars_data.find().limit(limit))
        return list(self.cars_data.find())
    
    def get_statistics(self):
        """Get database statistics"""
        return {
            'total_predictions': self.predictions.count_documents({}),
            'total_cars': self.cars_data.count_documents({}),
            'latest_prediction': self.predictions.find_one(sort=[('timestamp', -1)])
        }