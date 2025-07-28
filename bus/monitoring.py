"""
Monitoring and Analytics System for Syrian Bus Route Assistant

This module provides comprehensive monitoring, analytics, and performance tracking
for the bus route application.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from django.core.cache import cache
from django.conf import settings
from django.db import connection
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import psutil
import os

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor application performance and system metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def record_request(self, endpoint: str, duration: float, status_code: int, 
                      user_agent: str = None, ip_address: str = None):
        """
        Record API request metrics.
        
        Args:
            endpoint: API endpoint called
            duration: Request duration in seconds
            status_code: HTTP status code
            user_agent: User agent string
            ip_address: Client IP address
        """
        self.request_count += 1
        
        if status_code >= 400:
            self.error_count += 1
        
        # Store request metrics in cache
        request_data = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'duration': duration,
            'status_code': status_code,
            'user_agent': user_agent,
            'ip_address': ip_address
        }
        
        # Store in cache for analytics
        cache_key = f"request_log:{datetime.now().strftime('%Y%m%d%H%M')}"
        requests = cache.get(cache_key, [])
        requests.append(request_data)
        cache.set(cache_key, requests, 3600)  # Keep for 1 hour
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        uptime = time.time() - self.start_time
        
        # Calculate cache hit rate
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
        
        # Calculate error rate
        error_rate = (self.error_count / self.request_count * 100) if self.request_count > 0 else 0
        
        # Get system metrics
        system_metrics = self._get_system_metrics()
        
        return {
            'uptime_seconds': round(uptime, 2),
            'uptime_formatted': str(timedelta(seconds=int(uptime))),
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'error_rate_percent': round(error_rate, 2),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate_percent': round(cache_hit_rate, 2),
            'requests_per_minute': self._calculate_requests_per_minute(),
            'system_metrics': system_metrics
        }
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            
            return {
                'cpu_percent': round(cpu_percent, 2),
                'memory_percent': round(memory.percent, 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_percent': round(disk.percent, 2),
                'disk_free_gb': round(disk.free / (1024**3), 2),
                'network_bytes_sent_mb': round(network.bytes_sent / (1024**2), 2),
                'network_bytes_recv_mb': round(network.bytes_recv / (1024**2), 2)
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_requests_per_minute(self) -> float:
        """Calculate requests per minute based on recent data."""
        try:
            current_time = datetime.now()
            minute_ago = current_time - timedelta(minutes=1)
            
            # Get requests from the last minute
            requests_last_minute = 0
            for i in range(2):  # Check current and previous minute
                check_time = current_time - timedelta(minutes=i)
                cache_key = f"request_log:{check_time.strftime('%Y%m%d%H%M')}"
                requests = cache.get(cache_key, [])
                requests_last_minute += len(requests)
            
            return round(requests_last_minute / 2, 2)  # Average over 2 minutes
        except Exception as e:
            logger.error(f"Error calculating requests per minute: {e}")
            return 0.0

class DatabaseMonitor:
    def __init__(self):
        from .mongo import initialize_mongo
        
        try:
            self.client, self.db, self.collection = initialize_mongo()
        except Exception as e:
            logger.warning(f"DatabaseMonitor initialization failed: {e}")
            self.client = None
            self.db = None
            self.collection = None
    
    def get_database_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive database metrics.
        
        Returns:
            Dictionary containing database metrics
        """
        try:
            if not self.client:
                return {'error': 'MongoDB not connected'}
            
            # Basic collection stats
            collection_stats = self.db.command('collStats', settings.MONGODB_COLLECTION)
            
            # Database stats
            db_stats = self.db.command('dbStats')
            
            # Connection info
            server_info = self.client.server_info()
            
            # Index usage
            index_stats = self.collection.aggregate([
                {'$indexStats': {}}
            ])
            index_usage = list(index_stats)
            
            return {
                'collection_stats': {
                    'document_count': collection_stats.get('count', 0),
                    'storage_size_mb': round(collection_stats.get('storageSize', 0) / (1024**2), 2),
                    'index_size_mb': round(collection_stats.get('totalIndexSize', 0) / (1024**2), 2),
                    'avg_document_size_bytes': round(collection_stats.get('avgObjSize', 0), 2)
                },
                'database_stats': {
                    'database_size_mb': round(db_stats.get('dataSize', 0) / (1024**2), 2),
                    'storage_size_mb': round(db_stats.get('storageSize', 0) / (1024**2), 2),
                    'index_size_mb': round(db_stats.get('indexSize', 0) / (1024**2), 2)
                },
                'server_info': {
                    'version': server_info.get('version', 'Unknown'),
                    'uptime_hours': round(server_info.get('uptime', 0) / 3600, 2),
                    'connections_current': server_info.get('connections', {}).get('current', 0),
                    'connections_available': server_info.get('connections', {}).get('available', 0)
                },
                'index_usage': index_usage
            }
        except Exception as e:
            logger.error(f"Error getting database metrics: {e}")
            return {'error': str(e)}
    
    def test_database_connection(self) -> Dict[str, Any]:
        """
        Test database connectivity and performance.
        
        Returns:
            Dictionary containing connection test results
        """
        try:
            if not self.client:
                return {'status': 'error', 'message': 'MongoDB not connected'}
            
            start_time = time.time()
            
            # Test basic query
            test_query = self.collection.find_one({})
            query_time = time.time() - start_time
            
            # Test count operation
            start_time = time.time()
            count = self.collection.count_documents({})
            count_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'connection_test': 'passed',
                'query_response_time_ms': round(query_time * 1000, 2),
                'count_response_time_ms': round(count_time * 1000, 2),
                'total_documents': count,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }

class AnalyticsCollector:
    """Collect and analyze usage analytics."""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.db_monitor = DatabaseMonitor()
    
    def collect_endpoint_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Collect analytics for API endpoints.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary containing endpoint analytics
        """
        try:
            current_time = datetime.now()
            endpoint_stats = {}
            
            # Collect data from cache for the specified hours
            for i in range(hours):
                check_time = current_time - timedelta(hours=i)
                cache_key = f"request_log:{check_time.strftime('%Y%m%d%H%M')}"
                requests = cache.get(cache_key, [])
                
                for request in requests:
                    endpoint = request.get('endpoint', 'unknown')
                    status_code = request.get('status_code', 0)
                    duration = request.get('duration', 0)
                    
                    if endpoint not in endpoint_stats:
                        endpoint_stats[endpoint] = {
                            'total_requests': 0,
                            'successful_requests': 0,
                            'error_requests': 0,
                            'total_duration': 0,
                            'avg_duration': 0,
                            'status_codes': {}
                        }
                    
                    stats = endpoint_stats[endpoint]
                    stats['total_requests'] += 1
                    stats['total_duration'] += duration
                    
                    if 200 <= status_code < 400:
                        stats['successful_requests'] += 1
                    else:
                        stats['error_requests'] += 1
                    
                    # Track status codes
                    status_str = str(status_code)
                    stats['status_codes'][status_str] = stats['status_codes'].get(status_str, 0) + 1
            
            # Calculate averages
            for endpoint, stats in endpoint_stats.items():
                if stats['total_requests'] > 0:
                    stats['avg_duration'] = round(stats['total_duration'] / stats['total_requests'], 3)
                    stats['success_rate'] = round(
                        (stats['successful_requests'] / stats['total_requests']) * 100, 2
                    )
            
            return {
                'period_hours': hours,
                'total_endpoints': len(endpoint_stats),
                'endpoint_stats': endpoint_stats,
                'timestamp': current_time.isoformat()
            }
        except Exception as e:
            logger.error(f"Error collecting endpoint analytics: {e}")
            return {'error': str(e)}
    
    def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics report.
        
        Returns:
            Dictionary containing comprehensive analytics
        """
        try:
            performance_metrics = self.monitor.get_performance_metrics()
            database_metrics = self.db_monitor.get_database_metrics()
            endpoint_analytics = self.collect_endpoint_analytics(hours=1)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'performance': performance_metrics,
                'database': database_metrics,
                'endpoints': endpoint_analytics,
                'system_health': self._assess_system_health(performance_metrics, database_metrics)
            }
        except Exception as e:
            logger.error(f"Error getting comprehensive analytics: {e}")
            return {'error': str(e)}
    
    def _assess_system_health(self, performance: Dict, database: Dict) -> Dict[str, str]:
        """
        Assess overall system health based on metrics.
        
        Args:
            performance: Performance metrics
            database: Database metrics
            
        Returns:
            Dictionary containing health assessments
        """
        health_status = {}
        
        # Performance health
        error_rate = performance.get('error_rate_percent', 0)
        if error_rate < 5:
            health_status['performance'] = 'excellent'
        elif error_rate < 10:
            health_status['performance'] = 'good'
        elif error_rate < 20:
            health_status['performance'] = 'warning'
        else:
            health_status['performance'] = 'critical'
        
        # Database health
        if 'error' not in database:
            health_status['database'] = 'healthy'
        else:
            health_status['database'] = 'error'
        
        # System resource health
        system_metrics = performance.get('system_metrics', {})
        cpu_percent = system_metrics.get('cpu_percent', 0)
        memory_percent = system_metrics.get('memory_percent', 0)
        
        if cpu_percent < 70 and memory_percent < 80:
            health_status['resources'] = 'healthy'
        elif cpu_percent < 90 and memory_percent < 90:
            health_status['resources'] = 'warning'
        else:
            health_status['resources'] = 'critical'
        
        # Overall health
        if all(status in ['excellent', 'good', 'healthy'] for status in health_status.values()):
            health_status['overall'] = 'healthy'
        elif any(status == 'critical' for status in health_status.values()):
            health_status['overall'] = 'critical'
        else:
            health_status['overall'] = 'warning'
        
        return health_status

# Global instances
performance_monitor = PerformanceMonitor()
database_monitor = DatabaseMonitor()
analytics_collector = AnalyticsCollector() 