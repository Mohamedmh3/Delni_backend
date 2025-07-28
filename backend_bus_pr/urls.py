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
    path('admin/', admin.site.urls),
    path('api/', include('bus.urls')),
    
    # API Documentation
    path('swagger<format>/', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]
