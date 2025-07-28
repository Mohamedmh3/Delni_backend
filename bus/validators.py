"""
Input validation and error handling utilities for the Syrian Bus Route Assistant.

This module provides comprehensive validation for:
- Coordinate validation (latitude/longitude ranges)
- Parameter type and range validation
- Custom exception classes for better error handling
- Validation decorators for API endpoints
"""

import re
from typing import Dict, Any, Tuple, Optional
from django.http import JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view
from functools import wraps


class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, field: str = None, code: str = None):
        self.message = message
        self.field = field
        self.code = code
        super().__init__(self.message)


class CoordinateValidationError(ValidationError):
    """Exception for coordinate validation errors."""
    pass


class ParameterValidationError(ValidationError):
    """Exception for parameter validation errors."""
    pass


def validate_coordinate(value: Any, field_name: str) -> float:
    """
    Validate and convert coordinate value to float.
    
    Args:
        value: The coordinate value to validate
        field_name: Name of the field for error messages
        
    Returns:
        float: Validated coordinate value
        
    Raises:
        CoordinateValidationError: If coordinate is invalid
    """
    try:
        coord = float(value)
    except (ValueError, TypeError):
        raise CoordinateValidationError(
            f"Invalid {field_name}: must be a valid number",
            field=field_name,
            code="INVALID_COORDINATE_TYPE"
        )
    
    # Validate latitude range (-90 to 90)
    if field_name.lower().endswith('lat'):
        if not -90 <= coord <= 90:
            raise CoordinateValidationError(
                f"Invalid {field_name}: latitude must be between -90 and 90 degrees",
                field=field_name,
                code="INVALID_LATITUDE_RANGE"
            )
    
    # Validate longitude range (-180 to 180)
    elif field_name.lower().endswith('lng'):
        if not -180 <= coord <= 180:
            raise CoordinateValidationError(
                f"Invalid {field_name}: longitude must be between -180 and 180 degrees",
                field=field_name,
                code="INVALID_LONGITUDE_RANGE"
            )
    
    return coord


def validate_coordinates(lat: float, lng: float) -> Tuple[float, float]:
    """
    Validate a pair of coordinates (latitude and longitude).
    
    Args:
        lat: Latitude value
        lng: Longitude value
        
    Returns:
        Tuple[float, float]: Validated (latitude, longitude) pair
        
    Raises:
        CoordinateValidationError: If coordinates are invalid
    """
    validated_lat = validate_coordinate(lat, 'latitude')
    validated_lng = validate_coordinate(lng, 'longitude')
    return validated_lat, validated_lng


def validate_required_parameters(request_params: Dict[str, Any], required_params: list) -> Dict[str, Any]:
    """
    Validate that all required parameters are present.
    
    Args:
        request_params: Dictionary of request parameters
        required_params: List of required parameter names
        
    Returns:
        Dict[str, Any]: Validated parameters
        
    Raises:
        ParameterValidationError: If required parameters are missing
    """
    missing_params = []
    validated_params = {}
    
    for param in required_params:
        if param not in request_params:
            missing_params.append(param)
        else:
            validated_params[param] = request_params[param]
    
    if missing_params:
        raise ParameterValidationError(
            f"Missing required parameters: {', '.join(missing_params)}",
            field="parameters",
            code="MISSING_REQUIRED_PARAMETERS"
        )
    
    return validated_params


def validate_optional_parameter(value: Any, field_name: str, expected_type: type, 
                               default_value: Any = None, valid_values: list = None) -> Any:
    """
    Validate optional parameter with type checking and value validation.
    
    Args:
        value: The parameter value to validate
        field_name: Name of the field for error messages
        expected_type: Expected data type
        default_value: Default value if parameter is None
        valid_values: List of valid values (if applicable)
        
    Returns:
        Any: Validated parameter value
        
    Raises:
        ParameterValidationError: If parameter is invalid
    """
    if value is None:
        return default_value
    
    # Type validation
    if not isinstance(value, expected_type):
        try:
            if expected_type == float:
                value = float(value)
            elif expected_type == int:
                value = int(value)
            elif expected_type == str:
                value = str(value)
        except (ValueError, TypeError):
            raise ParameterValidationError(
                f"Invalid {field_name}: expected {expected_type.__name__}, got {type(value).__name__}",
                field=field_name,
                code="INVALID_PARAMETER_TYPE"
            )
    
    # Value validation
    if valid_values is not None and value not in valid_values:
        raise ParameterValidationError(
            f"Invalid {field_name}: must be one of {valid_values}",
            field=field_name,
            code="INVALID_PARAMETER_VALUE"
        )
    
    return value


def validate_route_parameters(request_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive validation for route finding parameters.
    
    Args:
        request_params: Dictionary of request parameters
        
    Returns:
        Dict[str, Any]: Validated parameters
        
    Raises:
        ValidationError: If any validation fails
    """
    # Required parameters
    required_params = ['from_lng', 'from_lat', 'to_lng', 'to_lat']
    validated = validate_required_parameters(request_params, required_params)
    
    # Validate coordinates
    validated['from_lng'] = validate_coordinate(validated['from_lng'], 'from_lng')
    validated['from_lat'] = validate_coordinate(validated['from_lat'], 'from_lat')
    validated['to_lng'] = validate_coordinate(validated['to_lng'], 'to_lng')
    validated['to_lat'] = validate_coordinate(validated['to_lat'], 'to_lat')
    
    # Optional parameters - Västtrafik-style categories
    validated['category'] = validate_optional_parameter(
        request_params.get('category'),
        'category',
        str,
        default_value=None,
        valid_values=['fewest_walking', 'least_transfers', 'fastest']
    )
    
    # Validate that origin and destination are different
    origin = (validated['from_lng'], validated['from_lat'])
    destination = (validated['to_lng'], validated['to_lat'])
    
    if origin == destination:
        raise ValidationError(
            "Origin and destination cannot be the same",
            field="coordinates",
            code="SAME_ORIGIN_DESTINATION"
        )
    
    return validated


def handle_validation_error(error: ValidationError) -> JsonResponse:
    """
    Convert validation errors to consistent JSON responses.
    
    Args:
        error: ValidationError instance
        
    Returns:
        JsonResponse: Formatted error response
    """
    error_response = {
        "error": True,
        "message": error.message,
        "code": error.code or "VALIDATION_ERROR"
    }
    
    if error.field:
        error_response["field"] = error.field
    
    return JsonResponse(error_response, status=status.HTTP_400_BAD_REQUEST)


def validate_api_request(validation_func):
    """
    Decorator for API endpoints to add automatic validation.
    
    Args:
        validation_func: Function to validate request parameters
        
    Returns:
        Decorated function with validation
    """
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            try:
                # Validate parameters
                validated_params = validation_func(request.GET)
                
                # Add validated parameters to request
                request.validated_params = validated_params
                
                return view_func(request, *args, **kwargs)
                
            except ValidationError as e:
                return handle_validation_error(e)
            except Exception as e:
                # Log unexpected errors
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Unexpected error in {view_func.__name__}: {str(e)}")
                
                return JsonResponse({
                    "error": True,
                    "message": "Internal server error. Please try again later.",
                    "code": "INTERNAL_ERROR"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return wrapper
    return decorator


def validate_coordinates_in_syria(lng: float, lat: float) -> bool:
    """
    Validate that coordinates are within Syria's boundaries.
    
    Args:
        lng: Longitude coordinate
        lat: Latitude coordinate
        
    Returns:
        bool: True if coordinates are within Syria
    """
    # Syria's approximate boundaries
    SYRIA_BOUNDS = {
        'min_lng': 35.6,
        'max_lng': 42.4,
        'min_lat': 32.3,
        'max_lat': 37.3
    }
    
    return (SYRIA_BOUNDS['min_lng'] <= lng <= SYRIA_BOUNDS['max_lng'] and
            SYRIA_BOUNDS['min_lat'] <= lat <= SYRIA_BOUNDS['max_lat'])


def sanitize_string_input(value: str, max_length: int = 100) -> str:
    """
    Sanitize string input to prevent injection attacks.
    
    Args:
        value: String value to sanitize
        max_length: Maximum allowed length
        
    Returns:
        str: Sanitized string
        
    Raises:
        ValidationError: If string is invalid
    """
    if not isinstance(value, str):
        raise ValidationError("Input must be a string", code="INVALID_STRING_TYPE")
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', value.strip())
    
    if len(sanitized) > max_length:
        raise ValidationError(
            f"String too long. Maximum length is {max_length} characters",
            code="STRING_TOO_LONG"
        )
    
    return sanitized


def validate_pagination_parameters(request_params: Dict[str, Any]) -> Dict[str, int]:
    """
    Validate pagination parameters.
    
    Args:
        request_params: Dictionary of request parameters
        
    Returns:
        Dict[str, int]: Validated pagination parameters
    """
    page = validate_optional_parameter(
        request_params.get('page'),
        'page',
        int,
        default_value=1
    )
    
    page_size = validate_optional_parameter(
        request_params.get('page_size'),
        'page_size',
        int,
        default_value=10
    )
    
    # Validate ranges
    if page < 1:
        raise ParameterValidationError(
            "Page number must be greater than 0",
            field="page",
            code="INVALID_PAGE_NUMBER"
        )
    
    if page_size < 1 or page_size > 100:
        raise ParameterValidationError(
            "Page size must be between 1 and 100",
            field="page_size",
            code="INVALID_PAGE_SIZE"
        )
    
    return {
        'page': page,
        'page_size': page_size
    }


def validate_route_preferences(preferences: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate route preferences and optimization settings.
    
    Args:
        preferences: Dictionary of route preferences
        
    Returns:
        Dict[str, Any]: Validated preferences
        
    Raises:
        ParameterValidationError: If preferences are invalid
    """
    validated = {}
    
    # Category preference
    validated['category'] = validate_optional_parameter(
        preferences.get('category'),
        'category',
        str,
        default_value='default',
        valid_values=['default', 'fewest_walking', 'fewest_transfers', 'fastest']
    )
    
    # Max walking distance
    validated['max_walking_distance'] = validate_optional_parameter(
        preferences.get('max_walking_distance'),
        'max_walking_distance',
        float,
        default_value=1000.0,
        valid_values=None
    )
    
    # Max transfers
    validated['max_transfers'] = validate_optional_parameter(
        preferences.get('max_transfers'),
        'max_transfers',
        int,
        default_value=3,
        valid_values=None
    )
    
    # Validate ranges
    if validated['max_walking_distance'] < 0 or validated['max_walking_distance'] > 10000:
        raise ParameterValidationError(
            "Max walking distance must be between 0 and 10000 meters",
            field="max_walking_distance",
            code="INVALID_WALKING_DISTANCE"
        )
    
    if validated['max_transfers'] < 0 or validated['max_transfers'] > 10:
        raise ParameterValidationError(
            "Max transfers must be between 0 and 10",
            field="max_transfers",
            code="INVALID_MAX_TRANSFERS"
        )
    
    return validated 