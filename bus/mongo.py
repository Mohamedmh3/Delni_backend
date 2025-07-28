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
        logger.info(f"MONGO_URI: {settings.MONGO_URI[:20]}...")  # Log first 20 chars for debugging
        logger.info(f"MONGODB_DATABASE: {settings.MONGODB_DATABASE}")
        logger.info(f"MONGODB_COLLECTION: {settings.MONGODB_COLLECTION}")
        
        # Create MongoDB client with timeout
        mongo_client = MongoClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=10000,  # 10 seconds timeout
            connectTimeoutMS=10000,
            socketTimeoutMS=10000
        )
        
        # Test the connection by getting server info
        logger.info("Testing MongoDB connection...")
        server_info = mongo_client.server_info()
        logger.info(f"MongoDB server info: {server_info.get('version', 'Unknown version')}")
        
        # Initialize database and collection
        db = mongo_client[settings.MONGODB_DATABASE]
        collection = db[settings.MONGODB_COLLECTION]
        
        # Test database access
        collections = db.list_collection_names()
        logger.info(f"Available collections: {collections}")
        
        logger.info("MongoDB connection initialized successfully")
        return mongo_client, db, collection
        
    except ServerSelectionTimeoutError as e:
        logger.error(f"MongoDB server selection timeout: {e}")
        logger.error("This usually means Railway can't reach MongoDB Atlas")
        logger.error("Possible causes: IP whitelist, network restrictions, or wrong URI")
        mongo_client = None
        db = None
        collection = None
        raise
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failure: {e}")
        logger.error("This usually means authentication failed or network issues")
        mongo_client = None
        db = None
        collection = None
        raise
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB connection: {e}")
        logger.error(f"Error type: {type(e).__name__}")
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