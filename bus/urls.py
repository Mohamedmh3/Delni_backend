# bus_routes/urls.py
from django.urls import path
from .views import find_route, route_statistics, health_check, diagnostics, top_suggestions, suggest_route, graph_route, real_time_info
urlpatterns = [
    path('find-route/', find_route, name='find_route'),
    path('statistics/', route_statistics, name='route_statistics'),
    path('health/', health_check, name='health_check'),
    path('diagnostics/', diagnostics, name='diagnostics'),
    path('top-suggestions/', top_suggestions, name='top_suggestions'),
    path('suggest-route/', suggest_route, name='suggest_route'),
    path('graph-route/', graph_route, name='graph_route'),
    path('real-time-info/', real_time_info, name='real_time_info'),
]
