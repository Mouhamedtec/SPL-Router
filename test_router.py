#!/usr/bin/env python3
"""
Unit tests for the OSMnx routing engine.
"""

import unittest
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
import osmnx as ox
from typing import Tuple

# Add the current directory to the path to import the router module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router import OSMRoutingEngine


class TestOSMRoutingEngine(unittest.TestCase):
    """Test cases for the OSMRoutingEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock graph for testing
        self.mock_graph = nx.MultiDiGraph()
        
        # Add some test nodes
        self.mock_graph.add_node(1, x=-122.4194, y=37.7749)
        self.mock_graph.add_node(2, x=-122.4313, y=37.8051)
        self.mock_graph.add_node(3, x=-122.4250, y=37.7900)
        
        # Add some test edges
        self.mock_graph.add_edge(1, 2, length=1000.0)
        self.mock_graph.add_edge(2, 3, length=800.0)
        self.mock_graph.add_edge(1, 3, length=1500.0)
        
        # Test coordinates
        self.start_point = (-122.4194, 37.7749)
        self.end_point = (-122.4313, 37.8051)
        self.invalid_coords = (200.0, 100.0)  # Invalid coordinates

    def test_init_with_place_name(self):
        """Test initialization with place name."""
        with patch('osmnx.graph_from_place') as mock_graph_from_place:
            mock_graph_from_place.return_value = self.mock_graph
            
            with patch('osmnx.add_edge_speeds') as mock_add_speeds:
                mock_add_speeds.return_value = self.mock_graph
                
                with patch('osmnx.add_edge_travel_times') as mock_add_times:
                    mock_add_times.return_value = self.mock_graph
                    
                    router = OSMRoutingEngine(place_name="Test City")
                    
                    self.assertIsNotNone(router.graph)
                    self.assertEqual(router.place_name, "Test City")


    def test_init_with_no_parameters(self):
        """Test initialization with no parameters."""
        with self.assertRaises(ValueError):
            OSMRoutingEngine()

    def test_validate_coordinates_valid(self):
        """Test coordinate validation with valid coordinates."""
        router = OSMRoutingEngine(place_name="Test")
        router.graph = self.mock_graph
        
        # Test valid coordinates
        self.assertTrue(router.validate_coordinates(-122.4194, 37.7749))
        self.assertTrue(router.validate_coordinates(0.0, 0.0))
        self.assertTrue(router.validate_coordinates(180.0, 90.0))
        self.assertTrue(router.validate_coordinates(-180.0, -90.0))

    def test_validate_coordinates_invalid(self):
        """Test coordinate validation with invalid coordinates."""
        router = OSMRoutingEngine(place_name="Test")
        router.graph = self.mock_graph
        
        # Test invalid coordinates
        self.assertFalse(router.validate_coordinates(200.0, 37.7749))  # Longitude > 180
        self.assertFalse(router.validate_coordinates(-200.0, 37.7749))  # Longitude < -180
        self.assertFalse(router.validate_coordinates(-122.4194, 100.0))  # Latitude > 90
        self.assertFalse(router.validate_coordinates(-122.4194, -100.0))  # Latitude < -90

    def test_haversine_distance(self):
        """Test haversine distance calculation."""
        router = OSMRoutingEngine(place_name="Test")
        router.graph = self.mock_graph
        
        # Test distance calculation
        distance = router.haversine_distance(37.7749, -122.4194, 37.8051, -122.4313)
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0)

    def test_shortest_path_success(self):
        """Test successful shortest path calculation."""
        router = OSMRoutingEngine(place_name="Test")
        router.graph = self.mock_graph
        
        # Mock the nearest_nodes function
        with patch('osmnx.distance.nearest_nodes') as mock_nearest:
            mock_nearest.side_effect = [1, 2]  # Return node IDs 1 and 2
            
            path_coords, distance = router.shortest_path(self.start_point, self.end_point)
            
            self.assertIsInstance(path_coords, list)
            self.assertIsInstance(distance, float)
            self.assertGreater(len(path_coords), 0)

    def test_shortest_path_same_start_end(self):
        """Test shortest path with same start and end points."""
        router = OSMRoutingEngine(place_name="Test")
        router.graph = self.mock_graph
        
        with patch('osmnx.distance.nearest_nodes') as mock_nearest:
            mock_nearest.return_value = 1  # Same node for start and end
            
            path_coords, distance = router.shortest_path(self.start_point, self.start_point)
            
            self.assertEqual(len(path_coords), 1)
            self.assertEqual(distance, 0.0)

    def test_shortest_path_invalid_coordinates(self):
        """Test shortest path with invalid coordinates."""
        router = OSMRoutingEngine(place_name="Test")
        router.graph = self.mock_graph
        
        with self.assertRaises(ValueError):
            router.shortest_path(self.invalid_coords, self.end_point)
        
        with self.assertRaises(ValueError):
            router.shortest_path(self.start_point, self.invalid_coords)

    def test_shortest_path_no_graph(self):
        """Test shortest path with no graph loaded."""
        router = OSMRoutingEngine(place_name="Test")
        router.graph = None
        
        with self.assertRaises(RuntimeError):
            router.shortest_path(self.start_point, self.end_point)

    def test_shortest_path_empty_graph(self):
        """Test shortest path with empty graph."""
        router = OSMRoutingEngine(place_name="Test")
        router.graph = nx.MultiDiGraph()  # Empty graph
        
        with self.assertRaises(RuntimeError):
            router.shortest_path(self.start_point, self.end_point)

    def test_get_graph_info(self):
        """Test graph info retrieval."""
        router = OSMRoutingEngine(place_name="Test")
        router.graph = self.mock_graph
        
        info = router.get_graph_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('nodes', info)
        self.assertIn('edges', info)
        self.assertIn('is_projected', info)
        self.assertIn('crs', info)
        self.assertEqual(info['nodes'], 3)
        self.assertEqual(info['edges'], 3)

    def test_get_graph_info_no_graph(self):
        """Test graph info retrieval with no graph."""
        router = OSMRoutingEngine(place_name="Test")
        router.graph = None
        
        info = router.get_graph_info()
        self.assertEqual(info, {"error": "Graph not loaded"})

    @patch('matplotlib.pyplot')
    def test_visualize_route(self, mock_plt):
        """Test route visualization."""
        router = OSMRoutingEngine(place_name="Test")
        router.graph = self.mock_graph
        
        with patch('osmnx.distance.nearest_nodes') as mock_nearest:
            mock_nearest.side_effect = [1, 2]
            
            # Test visualization without saving
            router.visualize_route(self.start_point, self.end_point)
            
            # Test visualization with saving
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_file_path = temp_file.name
            
            try:
                router.visualize_route(self.start_point, self.end_point, save_path=temp_file_path)
                self.assertTrue(os.path.exists(temp_file_path))
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

    @patch('folium.Map')
    def test_visualize_route_interactive(self, mock_folium_map):
        """Test interactive route visualization."""
        router = OSMRoutingEngine(place_name="Test")
        router.graph = self.mock_graph
        
        with patch('osmnx.distance.nearest_nodes') as mock_nearest:
            mock_nearest.side_effect = [1, 2]
            
            # Mock folium components
            mock_map = MagicMock()
            mock_folium_map.return_value = mock_map
            
            router.visualize_route_interactive(self.start_point, self.end_point)
            
            # Check if save was called
            mock_map.save.assert_called_once()

    @patch('matplotlib.pyplot')
    def test_plot_route_stats(self, mock_plt):
        """Test route statistics plotting."""
        router = OSMRoutingEngine(place_name="Test")
        router.graph = self.mock_graph
        
        with patch('osmnx.distance.nearest_nodes') as mock_nearest:
            mock_nearest.side_effect = [1, 2]
            
            router.plot_route_stats(self.start_point, self.end_point)
            
            # Check if matplotlib was used
            mock_plt.subplots.assert_called_once()

    def test_load_graph_failure(self):
        """Test graph loading failure."""
        with patch('osmnx.graph_from_place') as mock_graph_from_place:
            mock_graph_from_place.side_effect = Exception("Network error")
            
            with self.assertRaises(RuntimeError):
                OSMRoutingEngine(place_name="Invalid Place")

    def test_edge_case_coordinates(self):
        """Test edge case coordinates."""
        router = OSMRoutingEngine(place_name="Test")
        router.graph = self.mock_graph
        
        # Test boundary coordinates
        self.assertTrue(router.validate_coordinates(180.0, 90.0))
        self.assertTrue(router.validate_coordinates(-180.0, -90.0))
        self.assertTrue(router.validate_coordinates(0.0, 0.0))
        
        # Test just outside boundary
        self.assertFalse(router.validate_coordinates(180.1, 90.0))
        self.assertFalse(router.validate_coordinates(-180.1, -90.0))
        self.assertFalse(router.validate_coordinates(0.0, 90.1))
        self.assertFalse(router.validate_coordinates(0.0, -90.1))


class TestOSMRoutingEngineIntegration(unittest.TestCase):
    """Integration tests for the OSMRoutingEngine class."""

    def setUp(self):
        """Set up mock graph for integration tests."""
        self.mock_graph = nx.MultiDiGraph()
        self.mock_graph.add_node(1, x=-122.4194, y=37.7749)
        self.mock_graph.add_node(2, x=-122.4313, y=37.8051)
        self.mock_graph.add_node(3, x=-122.4250, y=37.7900)
        self.mock_graph.add_edge(1, 2, length=1000.0)
        self.mock_graph.add_edge(2, 3, length=800.0)
        self.mock_graph.add_edge(1, 3, length=1500.0)

    @unittest.skip("Skip integration test - requires internet connection")
    def test_real_place_routing(self):
        """Test routing with a real place (requires internet)."""
        router = OSMRoutingEngine(place_name="San Francisco, California")
        
        start_point = (-122.4194, 37.7749)
        end_point = (-122.4313, 37.8051)
        
        path_coords, distance = router.shortest_path(start_point, end_point)
        
        self.assertIsInstance(path_coords, list)
        self.assertIsInstance(distance, float)
        self.assertGreater(len(path_coords), 0)
        self.assertGreater(distance, 0)

    def test_coordinate_validation_comprehensive(self):
        """Comprehensive test of coordinate validation."""
        router = OSMRoutingEngine(place_name="Test")
        router.graph = self.mock_graph
        
        # Test various coordinate combinations
        test_cases = [
            # (lon, lat, expected_valid)
            (0, 0, True),
            (180, 90, True),
            (-180, -90, True),
            (181, 0, False),
            (-181, 0, False),
            (0, 91, False),
            (0, -91, False),
            (180.1, 0, False),
            (-180.1, 0, False),
            (0, 90.1, False),
            (0, -90.1, False),
        ]
        
        for lon, lat, expected in test_cases:
            with self.subTest(lon=lon, lat=lat):
                result = router.validate_coordinates(lon, lat)
                self.assertEqual(result, expected, f"Failed for coordinates ({lon}, {lat})")


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestOSMRoutingEngine))
    test_suite.addTest(unittest.makeSuite(TestOSMRoutingEngineIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 