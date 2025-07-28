"""
Custom Exceptions for Syrian Bus Route Assistant

This module defines custom exception classes for handling specific error scenarios
in the bus route application.
"""

from django.http import JsonResponse
from rest_framework import status


class BusRouteException(Exception):
    """Base exception class for bus route application."""
    
    def __init__(self, message: str, error_code: str = None, status_code: int = 400):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.status_code = status_code
        super().__init__(self.message)
    
    def to_response(self) -> JsonResponse:
        """Convert exception to JSON response."""
        return JsonResponse({
            'error': self.error_code,
            'message': self.message,
            'success': False
        }, status=self.status_code)


class InvalidCoordinatesError(BusRouteException):
    """Raised when coordinates are invalid or out of range."""
    
    def __init__(self, message: str = "Invalid coordinates provided"):
        super().__init__(message, "INVALID_COORDINATES", 400)


class NoRouteFoundError(BusRouteException):
    """Raised when no route is found between specified points."""
    
    def __init__(self, message: str = "No route found between the specified points"):
        super().__init__(message, "NO_ROUTE_FOUND", 404)


class ValidationError(BusRouteException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str = "Validation error"):
        super().__init__(message, "VALIDATION_ERROR", 400)


class DatabaseConnectionError(BusRouteException):
    """Raised when database connection fails."""
    
    def __init__(self, message: str = "Database connection failed"):
        super().__init__(message, "DATABASE_CONNECTION_ERROR", 503)


class ServiceUnavailableError(BusRouteException):
    """Raised when a service is temporarily unavailable."""
    
    def __init__(self, message: str = "Service temporarily unavailable"):
        super().__init__(message, "SERVICE_UNAVAILABLE", 503)


class RateLimitExceededError(BusRouteException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, "RATE_LIMIT_EXCEEDED", 429)


class ConfigurationError(BusRouteException):
    """Raised when there's a configuration error."""
    
    def __init__(self, message: str = "Configuration error"):
        super().__init__(message, "CONFIGURATION_ERROR", 500)


class GeographicBoundaryError(BusRouteException):
    """Raised when coordinates are outside Syria's boundaries."""
    
    def __init__(self, message: str = "Coordinates outside Syria's boundaries"):
        super().__init__(message, "GEOGRAPHIC_BOUNDARY_ERROR", 400)


class RouteProcessingError(BusRouteException):
    """Raised when there's an error processing route data."""
    
    def __init__(self, message: str = "Error processing route data"):
        super().__init__(message, "ROUTE_PROCESSING_ERROR", 500)


class CacheError(BusRouteException):
    """Raised when there's a cache-related error."""
    
    def __init__(self, message: str = "Cache operation failed"):
        super().__init__(message, "CACHE_ERROR", 500)


# Exception handler function for Django REST Framework
def handle_bus_route_exception(exc, context):
    """
    Custom exception handler for bus route exceptions.
    
    Args:
        exc: The exception that was raised
        context: The context in which the exception was raised
        
    Returns:
        Response object with error details
    """
    if isinstance(exc, BusRouteException):
        return exc.to_response()
    
    # For other exceptions, return a generic error response
    return JsonResponse({
        'error': 'INTERNAL_SERVER_ERROR',
        'message': 'An unexpected error occurred',
        'success': False
    }, status=500) 