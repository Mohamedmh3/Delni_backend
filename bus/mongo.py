"""
MongoDB connection initialization for the bus application.
"""
from pymongo import MongoClient
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# Initialize MongoDB client
try:
    mongo_client = MongoClient(settings.MONGO_URI)
    db = mongo_client[settings.MONGODB_DATABASE]
    collection = db[settings.MONGODB_COLLECTION]
    
    # Test the connection
    mongo_client.admin.command('ping')
    logger.info("MongoDB connection initialized successfully")
    
except Exception as e:
    logger.error(f"Failed to initialize MongoDB connection: {e}")
    mongo_client = None
    db = None
    collection = None 