from flask import Flask, request, jsonify
from flask_cors import CORS
from config import Config
from database import Database
from ml_model import CarPricePredictor
import os

app = Flask(__name__)
CORS(app, origins=Config.CORS_ORIGINS)

# Initialize components
db = Database()
predictor = CarPricePredictor()

# Initialize ML model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'car_price_model.pkl')
encoders_path = os.path.join(os.path.dirname(__file__), 'models', 'label_encoders.pkl')

if not predictor.load_model(model_path, encoders_path):
    print("Training new model...")
    predictor.train_model()
    predictor.save_model(model_path, encoders_path)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Car Price Prediction API is running'
    })

@app.route('/api/predict', methods=['POST'])
def predict_price():
    """Predict car price endpoint"""
    try:
        car_data = request.get_json()
        
        # Validate required fields
        required_fields = ['brand', 'model', 'year', 'fuel_type', 'transmission', 'kms_driven', 'location', 'owner']
        for field in required_fields:
            if field not in car_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Convert numeric fields
        try:
            car_data['year'] = int(car_data['year'])
            car_data['kms_driven'] = int(car_data['kms_driven'])
            car_data['owner'] = int(car_data['owner'])
        except ValueError as e:
            return jsonify({'error': f'Invalid numeric value: {str(e)}'}), 400
        
        # Make prediction
        predicted_price, confidence_score = predictor.predict_price(car_data)
        
        # Save prediction to database
        db.save_prediction(car_data, float(predicted_price), float(confidence_score))
        
        return jsonify({
            'predicted_price': round(predicted_price, 2),
            'confidence_score': round(confidence_score, 3),
            'price_range': {
                'min': round(predicted_price * (1 - (1 - confidence_score)), 2),
                'max': round(predicted_price * (1 + (1 - confidence_score)), 2)
            },
            'currency': 'INR'
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/car-brands', methods=['GET'])
def get_car_brands():
    """Get available car brands and models"""
    return jsonify(Config.CAR_BRANDS)

@app.route('/api/options', methods=['GET'])
def get_options():
    """Get all available options for dropdowns"""
    return jsonify({
        'brands': Config.CAR_BRANDS,
        'fuel_types': Config.FUEL_TYPES,
        'transmissions': Config.TRANSMISSION_TYPES,
        'locations': Config.LOCATIONS
    })

@app.route('/api/predictions/recent', methods=['GET'])
def get_recent_predictions():
    """Get recent predictions"""
    try:
        limit = int(request.args.get('limit', 10))
        predictions = db.get_recent_predictions(limit)
        
        # Convert ObjectId to string for JSON serialization
        for prediction in predictions:
            prediction['_id'] = str(prediction['_id'])
        
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': f'Failed to fetch predictions: {str(e)}'}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get database statistics"""
    try:
        stats = db.get_statistics()
        if stats['latest_prediction']:
            stats['latest_prediction']['_id'] = str(stats['latest_prediction']['_id'])
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': f'Failed to fetch statistics: {str(e)}'}), 500

@app.route('/api/train-model', methods=['POST'])
def retrain_model():
    """Retrain the ML model"""
    try:
        mae, r2 = predictor.train_model()
        predictor.save_model(model_path, encoders_path)
        
        return jsonify({
            'message': 'Model retrained successfully',
            'performance': {
                'mean_absolute_error': round(mae, 2),
                'r2_score': round(r2, 3)
            }
        })
    except Exception as e:
        return jsonify({'error': f'Model training failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)