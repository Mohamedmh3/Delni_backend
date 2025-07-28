"""
Django management command for comprehensive system health checks.

Usage:
    python manage.py health_check
"""

from django.core.management.base import BaseCommand
from django.conf import settings
from bus.services import BusRouteService
import logging
import time

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Perform comprehensive system health checks'

    def add_arguments(self, parser):
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed health information',
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('🔍 Performing system health checks...')
        )
        
        checks_passed = 0
        total_checks = 0
        
        # Check 1: Django settings
        total_checks += 1
        try:
            if hasattr(settings, 'MONGO_URI'):
                self.stdout.write('✅ MongoDB URI configured')
                checks_passed += 1
            else:
                self.stdout.write(
                    self.style.WARNING('⚠️  MongoDB URI not configured')
                )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Django settings check failed: {e}')
            )
        
        # Check 2: MongoDB connection
        total_checks += 1
        try:
            service = BusRouteService()
            # Test connection by getting database stats
            stats = service.get_database_stats()
            if 'error' not in stats:
                self.stdout.write('✅ MongoDB connection successful')
                checks_passed += 1
                
                if options['verbose']:
                    self.stdout.write(f"   Database: {stats.get('database_name')}")
                    self.stdout.write(f"   Collection: {stats.get('collection_name')}")
                    self.stdout.write(f"   Total routes: {stats.get('total_routes', 0)}")
            else:
                self.stdout.write(
                    self.style.ERROR(f'❌ MongoDB connection failed: {stats["error"]}')
                )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ MongoDB connection check failed: {e}')
            )
        
        # Check 3: API endpoints (basic test)
        total_checks += 1
        try:
            from django.test import Client
            client = Client()
            response = client.get('/api/health/')
            if response.status_code == 200:
                self.stdout.write('✅ API health endpoint accessible')
                checks_passed += 1
            else:
                self.stdout.write(
                    self.style.WARNING(f'⚠️  API health endpoint returned status {response.status_code}')
                )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ API health check failed: {e}')
            )
        
        # Check 4: Performance test
        total_checks += 1
        try:
            start_time = time.time()
            # Test a simple route query
            test_routes = service.get_all_routes()
            query_time = time.time() - start_time
            
            if query_time < 1.0:  # Should complete within 1 second
                self.stdout.write(f'✅ Performance test passed ({query_time:.3f}s)')
                checks_passed += 1
            else:
                self.stdout.write(
                    self.style.WARNING(f'⚠️  Performance test slow ({query_time:.3f}s)')
                )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Performance test failed: {e}')
            )
        
        # Check 5: Configuration validation
        total_checks += 1
        try:
            required_settings = [
                'MONGO_URI',
                'MONGODB_DATABASE', 
                'MONGODB_COLLECTION',
                'TAXI_SUGGESTION_DISTANCE'
            ]
            
            missing_settings = []
            for setting in required_settings:
                if not hasattr(settings, setting):
                    missing_settings.append(setting)
            
            if not missing_settings:
                self.stdout.write('✅ All required settings configured')
                checks_passed += 1
            else:
                self.stdout.write(
                    self.style.WARNING(f'⚠️  Missing settings: {", ".join(missing_settings)}')
                )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Configuration validation failed: {e}')
            )
        
        # Summary
        self.stdout.write('\n' + '='*50)
        self.stdout.write(f'Health Check Summary: {checks_passed}/{total_checks} checks passed')
        
        if checks_passed == total_checks:
            self.stdout.write(
                self.style.SUCCESS('🎉 All health checks passed! System is healthy.')
            )
            return 0
        else:
            self.stdout.write(
                self.style.WARNING(f'⚠️  {total_checks - checks_passed} check(s) failed. Review the issues above.')
            )
            return 1 