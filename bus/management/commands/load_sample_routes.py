"""
Django management command to load sample bus routes into MongoDB
Usage: python manage.py load_sample_routes
"""
from django.core.management.base import BaseCommand
from pymongo import MongoClient
from django.conf import settings
import json

class Command(BaseCommand):
    help = 'Load sample bus routes into MongoDB for testing'

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing data before loading sample routes',
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('Loading sample bus routes...')
        )
        
        # Connect to MongoDB
        client = MongoClient(settings.MONGODB_URI)
        db = client[settings.MONGODB_DATABASE]
        collection = db[settings.MONGODB_COLLECTION]
        
        # Sample bus routes data (Damascus area coordinates)
        sample_routes = [
            {
                "line_id": "Route-001",
                "route_name": "Damascus Central - Al Mazzeh",
                "route": {
                    "coordinates": [
                        [36.2021, 37.1343],  # Damascus Central
                        [36.2150, 37.1280],
                        [36.2300, 37.1150],
                        [36.2450, 37.1050],
                        [36.2600, 37.0950],
                        [36.2765, 37.0841]   # Al Mazzeh
                    ]
                },
                "schedule": {
                    "frequency": "every 15 minutes",
                    "operating_hours": "6:00 AM - 10:00 PM"
                },
                "description": "Main route connecting Damascus city center to Al Mazzeh district"
            },
            {
                "line_id": "Route-002", 
                "route_name": "Al Abbaseen - Kafr Souseh",
                "route": {
                    "coordinates": [
                        [36.3120, 37.0760],  # Al Abbaseen
                        [36.2980, 37.0820],
                        [36.2840, 37.0880],
                        [36.2700, 37.0940],
                        [36.2560, 37.1000],
                        [36.2420, 37.1060],
                        [36.2280, 37.1120],
                        [36.2140, 37.1180],
                        [36.2000, 37.1240]   # Kafr Souseh
                    ]
                },
                "schedule": {
                    "frequency": "every 20 minutes",
                    "operating_hours": "5:30 AM - 11:00 PM"
                },
                "description": "Route connecting Al Abbaseen area to Kafr Souseh"
            },
            {
                "line_id": "Route-003",
                "route_name": "Bab Touma - Mezzeh",
                "route": {
                    "coordinates": [
                        [36.3180, 37.0680],  # Bab Touma
                        [36.3040, 37.0740],
                        [36.2900, 37.0800],
                        [36.2760, 37.0860],
                        [36.2620, 37.0920],
                        [36.2480, 37.0980],
                        [36.2340, 37.1040],
                        [36.2200, 37.1100],
                        [36.2060, 37.1160],
                        [36.1920, 37.1220]   # Mezzeh
                    ]
                },
                "schedule": {
                    "frequency": "every 25 minutes",
                    "operating_hours": "6:00 AM - 9:30 PM"
                },
                "description": "Route from Bab Touma to Mezzeh area"
            },
            {
                "line_id": "Route-004",
                "route_name": "Al Midan - Yarmouk",
                "route": {
                    "coordinates": [
                        [36.3080, 37.0600],  # Al Midan
                        [36.2940, 37.0660],
                        [36.2800, 37.0720],
                        [36.2660, 37.0780],
                        [36.2520, 37.0840],
                        [36.2380, 37.0900],
                        [36.2240, 37.0960],
                        [36.2100, 37.1020],
                        [36.1960, 37.1080],
                        [36.1820, 37.1140],
                        [36.1680, 37.1200]   # Yarmouk
                    ]
                },
                "schedule": {
                    "frequency": "every 30 minutes",
                    "operating_hours": "5:00 AM - 10:30 PM"
                },
                "description": "Route connecting Al Midan to Yarmouk district"
            },
            {
                "line_id": "Route-005",
                "route_name": "Old City - Qanawat",
                "route": {
                    "coordinates": [
                        [36.3060, 37.0520],  # Old City
                        [36.2920, 37.0580],
                        [36.2780, 37.0640],
                        [36.2640, 37.0700],
                        [36.2500, 37.0760],
                        [36.2360, 37.0820],
                        [36.2220, 37.0880],
                        [36.2080, 37.0940],
                        [36.1940, 37.1000],
                        [36.1800, 37.1060],
                        [36.1660, 37.1120],
                        [36.1520, 37.1180]   # Qanawat
                    ]
                },
                "schedule": {
                    "frequency": "every 35 minutes",
                    "operating_hours": "6:30 AM - 9:00 PM"
                },
                "description": "Route from Old City to Qanawat area"
            }
        ]
        
        try:
            # Clear existing data if requested
            if options['clear']:
                collection.delete_many({})
                self.stdout.write(
                    self.style.WARNING('Cleared existing data from collection')
                )
            
            # Insert sample routes
            result = collection.insert_many(sample_routes)
            self.stdout.write(
                self.style.SUCCESS(f'Successfully inserted {len(result.inserted_ids)} sample routes')
            )
            
            # Verify insertion
            count = collection.count_documents({})
            self.stdout.write(
                self.style.SUCCESS(f'Total routes in database: {count}')
            )
            
            # Display sample route structure
            sample_route = collection.find_one({})
            if sample_route:
                self.stdout.write('\nSample route structure:')
                self.stdout.write(json.dumps(sample_route, indent=2, default=str))
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error creating sample routes: {e}')
            )
        
        finally:
            client.close()
            
        self.stdout.write(
            self.style.SUCCESS('Sample data loading completed!')
        ) 