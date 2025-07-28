from bson import ObjectId

def clean_mongo_document(doc):
    """
    Recursively convert ObjectId and other non-serializable types in a MongoDB document to strings.
    """
    if isinstance(doc, list):
        return [clean_mongo_document(item) for item in doc]
    elif isinstance(doc, dict):
        return {k: clean_mongo_document(v) for k, v in doc.items()}
    elif isinstance(doc, ObjectId):
        return str(doc)
    else:
        return doc 