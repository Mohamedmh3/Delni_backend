"""
MongoDB connection initialization for the bus application.
"""
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# Global variables
mongo_client = None
db = None
collection = None

def initialize_mongo():
    """Initialize MongoDB connection with proper error handling."""
    global mongo_client, db, collection
    
    if mongo_client is not None:
        return mongo_client, db, collection
    
    try:
        logger.info("Initializing MongoDB connection...")
        
        # Create MongoDB client with timeout
        mongo_client = MongoClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=10000,  # 10 seconds timeout
            connectTimeoutMS=10000,
            socketTimeoutMS=10000
        )
        
        # Test the connection by getting server info
        logger.info("Testing MongoDB connection...")
        mongo_client.server_info()
        
        # Initialize database and collection
        db = mongo_client[settings.MONGODB_DATABASE]
        collection = db[settings.MONGODB_COLLECTION]
        
        # Test database access
        db.list_collection_names()
        
        logger.info("MongoDB connection initialized successfully")
        return mongo_client, db, collection
        
    except ServerSelectionTimeoutError as e:
        logger.error(f"MongoDB server selection timeout: {e}")
        mongo_client = None
        db = None
        collection = None
        raise
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failure: {e}")
        mongo_client = None
        db = None
        collection = None
        raise
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB connection: {e}")
        mongo_client = None
        db = None
        collection = None
        raise

# Try to initialize on module import
try:
    initialize_mongo()
except Exception as e:
    logger.warning(f"MongoDB initialization failed on import: {e}")
    # Don't raise here, let the application continue 