"""
URL configuration for backend_bus_pr project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse
from django.conf import settings
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework import permissions

def root_view(request):
    """Simple root view for health checking"""
    return JsonResponse({
        'status': 'ok',
        'message': 'DELNI Backend API is running',
        'version': '1.0.0',
        'endpoints': {
            'api': '/api/',
            'health': '/api/health/',
            'swagger': '/swagger/',
            'admin': '/admin/'
        }
    })

def debug_settings(request):
    """Debug endpoint to check settings"""
    return JsonResponse({
        'status': 'debug',
        'settings': {
            'MONGO_URI': hasattr(settings, 'MONGO_URI'),
            'MONGO_URI_value': getattr(settings, 'MONGO_URI', 'NOT_SET'),
            'MONGODB_URI': hasattr(settings, 'MONGODB_URI'),
            'MONGODB_DATABASE': hasattr(settings, 'MONGODB_DATABASE'),
            'MONGODB_COLLECTION': hasattr(settings, 'MONGODB_COLLECTION'),
            'DEBUG': getattr(settings, 'DEBUG', 'NOT_SET'),
        }
    })

def test_mongo_connection(request):
    """Test endpoint to diagnose MongoDB connection issues."""
    from django.conf import settings
    from pymongo import MongoClient
    from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Test basic connection
        logger.info("Testing MongoDB connection from test endpoint...")
        
        client = MongoClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=5000,  # 5 seconds for quick test
            connectTimeoutMS=5000,
            socketTimeoutMS=5000
        )
        
        # Test server info
        server_info = client.server_info()
        
        # Test database access
        db = client[settings.MONGODB_DATABASE]
        collections = db.list_collection_names()
        
        client.close()
        
        return JsonResponse({
            'status': 'success',
            'message': 'MongoDB connection test successful',
            'server_version': server_info.get('version', 'Unknown'),
            'database': settings.MONGODB_DATABASE,
            'collections': collections,
            'uri_preview': settings.MONGO_URI[:50] + '...' if len(settings.MONGO_URI) > 50 else settings.MONGO_URI
        })
        
    except ServerSelectionTimeoutError as e:
        logger.error(f"MongoDB timeout in test: {e}")
        return JsonResponse({
            'status': 'error',
            'error': 'MongoDB connection timeout',
            'details': str(e),
            'suggestion': 'Check IP whitelist in MongoDB Atlas or network restrictions'
        }, status=500)
        
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failure in test: {e}")
        return JsonResponse({
            'status': 'error',
            'error': 'MongoDB connection failure',
            'details': str(e),
            'suggestion': 'Check username/password or network connectivity'
        }, status=500)
        
    except Exception as e:
        logger.error(f"Unexpected error in MongoDB test: {e}")
        return JsonResponse({
            'status': 'error',
            'error': 'Unexpected error',
            'details': str(e),
            'error_type': type(e).__name__
        }, status=500)

# Swagger/OpenAPI schema view
schema_view = get_schema_view(
    openapi.Info(
        title="Syrian Bus Route Assistant API",
        default_version='v1',
        description="""
        Comprehensive API for finding bus routes in Syria.
        
        ## Features
        - Find direct and multi-leg bus routes
        - Route optimization (fewest walking, fewest transfers)
        - Real-time route suggestions
        - Geographic coordinate validation
        
        ## Authentication
        This API is currently public and does not require authentication.
        
        ## Rate Limiting
        Please be respectful of the service and avoid excessive requests.
        """,
        terms_of_service="https://www.example.com/terms/",
        contact=openapi.Contact(email="contact@example.com"),
        license=openapi.License(name="MIT License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    path('', root_view, name='root'),
    path('debug/', debug_settings, name='debug'),
    path('test-mongo/', test_mongo_connection, name='test_mongo'),
    path('admin/', admin.site.urls),
    path('api/', include('bus.urls')),
    
    # API Documentation
    path('swagger<format>/', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]
