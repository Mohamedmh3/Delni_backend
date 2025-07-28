"""
Django management command to set up MongoDB indexes for optimal performance.

Usage:
    python manage.py setup_indexes
"""

from django.core.management.base import BaseCommand
from bus.services import BusRouteService
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Set up MongoDB indexes for optimal performance'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force recreation of existing indexes',
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('Setting up MongoDB indexes...')
        )
        
        try:
            service = BusRouteService()
            
            if options['force']:
                self.stdout.write('Dropping existing indexes...')
                service.collection.drop_indexes()
            
            # Create indexes
            service.create_indexes()
            
            # Get database stats
            stats = service.get_database_stats()
            
            self.stdout.write(
                self.style.SUCCESS('✅ MongoDB indexes created successfully!')
            )
            
            self.stdout.write(f"Database: {stats.get('database_name', 'Unknown')}")
            self.stdout.write(f"Collection: {stats.get('collection_name', 'Unknown')}")
            self.stdout.write(f"Total routes: {stats.get('total_routes', 0)}")
            
            if 'routes_by_type' in stats:
                self.stdout.write("Routes by type:")
                for route_type, count in stats['routes_by_type'].items():
                    self.stdout.write(f"  - {route_type}: {count}")
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error setting up indexes: {e}')
            )
            logger.error(f"Error in setup_indexes command: {e}")
            raise 