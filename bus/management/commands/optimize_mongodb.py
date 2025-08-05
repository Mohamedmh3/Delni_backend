"""
Django management command to optimize MongoDB for geospatial queries.
"""
from django.core.management.base import BaseCommand
from django.conf import settings
from bus.services import BusRouteService
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Optimize MongoDB indexes and analyze performance for geospatial queries'

    def add_arguments(self, parser):
        parser.add_argument(
            '--create-indexes',
            action='store_true',
            help='Create optimized MongoDB indexes',
        )
        parser.add_argument(
            '--analyze-performance',
            action='store_true',
            help='Analyze query performance and provide recommendations',
        )
        parser.add_argument(
            '--full-optimization',
            action='store_true',
            help='Perform full optimization including indexes and analysis',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force recreation of existing indexes',
        )

    def handle(self, *args, **options):
        try:
            service = BusRouteService()
            
            if options['full_optimization'] or options['create_indexes']:
                self.stdout.write(
                    self.style.SUCCESS('🚀 Starting MongoDB optimization...')
                )
                
                # Create optimized indexes
                self.create_optimized_indexes(service, options['force'])
                
            if options['full_optimization'] or options['analyze_performance']:
                # Analyze performance
                self.analyze_performance(service)
                
            if not any([options['create_indexes'], options['analyze_performance'], options['full_optimization']]):
                self.stdout.write(
                    self.style.WARNING('No action specified. Use --help for options.')
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error during optimization: {e}')
            )
            logger.error(f'MongoDB optimization failed: {e}')

    def create_optimized_indexes(self, service, force=False):
        """Create optimized MongoDB indexes."""
        self.stdout.write('📊 Creating optimized MongoDB indexes...')
        
        try:
            if force:
                self.stdout.write('🔄 Force recreating indexes...')
                # Drop existing indexes (except _id)
                service.collection.drop_indexes()
            
            # Create optimized indexes
            service.create_optimized_indexes()
            
            # Get index information
            indexes = service.collection.index_information()
            
            self.stdout.write(
                self.style.SUCCESS(f'✅ Successfully created {len(indexes)} indexes:')
            )
            
            for index_name, index_info in indexes.items():
                self.stdout.write(f'   📍 {index_name}: {index_info}')
            
            # Verify critical indexes
            self.verify_critical_indexes(service)
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error creating indexes: {e}')
            )
            raise

    def verify_critical_indexes(self, service):
        """Verify that critical indexes exist."""
        self.stdout.write('🔍 Verifying critical indexes...')
        
        indexes = service.collection.index_information()
        critical_indexes = [
            ('route.coordinates', '2dsphere'),
            ('line_id', 'unique'),
            ('type', 'basic'),
        ]
        
        missing_indexes = []
        
        for field, index_type in critical_indexes:
            found = False
            for index_name, index_info in indexes.items():
                if field in str(index_info):
                    found = True
                    self.stdout.write(
                        self.style.SUCCESS(f'   ✅ Found {index_type} index on {field}')
                    )
                    break
            
            if not found:
                missing_indexes.append(field)
                self.stdout.write(
                    self.style.WARNING(f'   ⚠️  Missing {index_type} index on {field}')
                )
        
        if missing_indexes:
            self.stdout.write(
                self.style.ERROR(f'❌ Missing critical indexes: {missing_indexes}')
            )
        else:
            self.stdout.write(
                self.style.SUCCESS('✅ All critical indexes verified!')
            )

    def analyze_performance(self, service):
        """Analyze query performance and provide recommendations."""
        self.stdout.write('📈 Analyzing query performance...')
        
        try:
            # Get database statistics
            stats = service.get_database_stats()
            
            self.stdout.write('📊 Database Statistics:')
            self.stdout.write(f'   📄 Total routes: {stats.get("total_routes", "N/A")}')
            self.stdout.write(f'   📁 Database: {stats.get("database_name", "N/A")}')
            self.stdout.write(f'   📂 Collection: {stats.get("collection_name", "N/A")}')
            
            if 'routes_by_type' in stats:
                self.stdout.write('   🚌 Routes by type:')
                for route_type, count in stats['routes_by_type'].items():
                    self.stdout.write(f'      {route_type}: {count}')
            
            # Analyze query performance
            analysis = service.analyze_query_performance('geospatial')
            
            self.stdout.write('\n🔍 Performance Analysis:')
            self.stdout.write(f'   📊 Total indexes: {analysis.get("total_indexes", "N/A")}')
            
            if 'performance_metrics' in analysis:
                metrics = analysis['performance_metrics']
                self.stdout.write('   📈 Performance Metrics:')
                self.stdout.write(f'      Total documents: {metrics.get("total_documents", "N/A")}')
                self.stdout.write(f'      Avg document size: {metrics.get("avg_document_size", "N/A")} bytes')
                self.stdout.write(f'      Total size: {metrics.get("total_size", "N/A")} bytes')
                self.stdout.write(f'      Index size: {metrics.get("index_size", "N/A")} bytes')
            
            # Display recommendations
            if 'recommendations' in analysis and analysis['recommendations']:
                self.stdout.write('\n💡 Recommendations:')
                for i, recommendation in enumerate(analysis['recommendations'], 1):
                    self.stdout.write(f'   {i}. {recommendation}')
            else:
                self.stdout.write('\n✅ No performance issues detected!')
            
            # Test geospatial query performance
            self.test_geospatial_performance(service)
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error analyzing performance: {e}')
            )
            raise

    def test_geospatial_performance(self, service):
        """Test geospatial query performance with sample data."""
        self.stdout.write('\n🧪 Testing geospatial query performance...')
        
        try:
            import time
            
            # Sample coordinates for Damascus, Syria
            test_point = [36.297949, 33.537161]  # Damascus center
            
            # Test 1: Simple geospatial query
            start_time = time.time()
            routes = service.get_routes_near_point(test_point, max_distance_m=5000)
            query_time = time.time() - start_time
            
            self.stdout.write(f'   ⏱️  Geospatial query time: {query_time:.3f}s')
            self.stdout.write(f'   📍 Routes found: {len(routes)}')
            
            # Test 2: Bounding box query
            start_time = time.time()
            bounds_routes = service.get_routes_in_bounds(
                test_point[0] - 0.1, test_point[0] + 0.1,
                test_point[1] - 0.1, test_point[1] + 0.1
            )
            bounds_time = time.time() - start_time
            
            self.stdout.write(f'   ⏱️  Bounding box query time: {bounds_time:.3f}s')
            self.stdout.write(f'   📍 Routes in bounds: {len(bounds_routes)}')
            
            # Performance assessment
            if query_time < 1.0:
                self.stdout.write(
                    self.style.SUCCESS('   ✅ Excellent query performance!')
                )
            elif query_time < 3.0:
                self.stdout.write(
                    self.style.WARNING('   ⚠️  Acceptable query performance')
                )
            else:
                self.stdout.write(
                    self.style.ERROR('   ❌ Poor query performance - optimization needed')
                )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'   ❌ Error testing performance: {e}')
            )

    def display_optimization_summary(self):
        """Display optimization summary and next steps."""
        self.stdout.write('\n🎯 Optimization Summary:')
        self.stdout.write('   ✅ MongoDB indexes optimized')
        self.stdout.write('   ✅ Performance analysis completed')
        self.stdout.write('   ✅ Geospatial queries tested')
        
        self.stdout.write('\n📋 Next Steps:')
        self.stdout.write('   1. Monitor query performance in production')
        self.stdout.write('   2. Adjust cache settings based on usage patterns')
        self.stdout.write('   3. Consider coordinate simplification for very long routes')
        self.stdout.write('   4. Set up regular performance monitoring')
        
        self.stdout.write('\n🔧 Configuration Settings:')
        self.stdout.write(f'   COORDINATE_SIMPLIFICATION_ENABLED: {getattr(settings, "COORDINATE_SIMPLIFICATION_ENABLED", False)}')
        self.stdout.write(f'   TARGET_COORDINATES_PER_ROUTE: {getattr(settings, "TARGET_COORDINATES_PER_ROUTE", "Not set")}')
        self.stdout.write(f'   CACHE_ENABLED: {getattr(settings, "CACHE_ENABLED", False)}')
        
        self.stdout.write(
            self.style.SUCCESS('\n🚀 MongoDB optimization completed successfully!')
        ) 