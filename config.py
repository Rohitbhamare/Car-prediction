import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'car_price_db')
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'car_price_model.pkl')
    PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), 'models', 'preprocessor.pkl')
    
    # CORS settings
    CORS_ORIGINS = ["http://localhost:5173", "https://localhost:5173", "http://localhost:3000"]
    
    # Car brands and models mapping
    CAR_BRANDS = {
        'Maruti': ['Swift', 'Baleno', 'WagonR', 'Alto', 'Vitara Brezza', 'Dzire', 'Ciaz'],
        'Hyundai': ['i20', 'Creta', 'Verna', 'Elite i20', 'Grand i10', 'Elantra', 'Tucson'],
        'Honda': ['City', 'Amaze', 'Jazz', 'WR-V', 'BR-V', 'Civic', 'CR-V'],
        'Toyota': ['Innova', 'Fortuner', 'Corolla', 'Etios', 'Yaris', 'Camry', 'Land Cruiser'],
        'Tata': ['Nexon', 'Harrier', 'Safari', 'Tiago', 'Tigor', 'Altroz', 'Punch'],
        'Mahindra': ['XUV500', 'Scorpio', 'Bolero', 'XUV300', 'Thar', 'KUV100', 'Marazzo'],
        'Ford': ['EcoSport', 'Endeavour', 'Figo', 'Aspire', 'Freestyle', 'Mustang'],
        'Renault': ['Duster', 'Kwid', 'Captur', 'Lodgy', 'Pulse', 'Scala'],
        'Volkswagen': ['Polo', 'Vento', 'Ameo', 'Tiguan', 'Passat', 'Jetta'],
        'Skoda': ['Rapid', 'Octavia', 'Superb', 'Kodiaq', 'Karoq', 'Kushaq']
    }
    
    FUEL_TYPES = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
    TRANSMISSION_TYPES = ['Manual', 'Automatic']
    LOCATIONS = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad', 'Surat', 'Jaipur']