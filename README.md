# SPL-Router (Routing Engine)

A robust and feature-rich routing engine built on top of OSMnx for OpenStreetMap data processing and route calculation. This project provides a simple yet powerful interface for finding optimal routes between geographical points with comprehensive visualization capabilities.

## 🚀 Features

- **Flexible Data Sources**: Support for both place names (automatic download) and OSM XML files
- **Robust Error Handling**: Comprehensive validation and error handling for edge cases
- **Multiple Visualization Options**: Static maps, interactive HTML maps, and route statistics
- **Coordinate Validation**: Built-in validation for geographical coordinates
- **Distance Calculations**: Accurate distance calculations using haversine formula
- **Graph Projection**: Automatic handling of coordinate reference systems
- **Comprehensive Testing**: Full test suite with unit and integration tests

## 📋 Requirements

### Core Dependencies
- Python 3.7+
- OSMnx >= 1.3.0
- NetworkX >= 2.8.0
- Pandas >= 1.5.0

### Visualization Dependencies
- Matplotlib >= 3.5.0
- Folium >= 0.12.0
- NumPy >= 1.21.0

### Testing Dependencies
- Pytest >= 7.0.0
- Pytest-cov >= 4.0.0
- Pytest-mock >= 3.8.0

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Mouhamedtec/SPL-Router.git
SPL-Router
```

### 2. Install Dependencies

#### Option A: Install all dependencies
```bash
pip install osmnx networkx pandas matplotlib folium numpy
```

#### Option B: Install with requirements file
```bash
pip install -r requirements.txt
```

#### Option C: Install for development (includes testing)
```bash
pip install -r requirements_test.txt
```

## 📖 Usage

### Basic Usage

```python
from router import OSMRoutingEngine

# Initialize with a place name (downloads data automatically)
router = OSMRoutingEngine(place_name="San Francisco, California")

# Define start and end points
start_point = (-122.4194, 37.7749)  # Union Square
end_point = (-122.4313, 37.8051)    # Fisherman's Wharf

# Find the shortest path
path_coords, distance = router.shortest_path(start_point, end_point)

print(f"Route found! Distance: {distance:.2f} meters")
print(f"Path has {len(path_coords)} coordinate points")
```

### Using OSM XML Files

```python
# Initialize with an OSM XML file
router = OSMRoutingEngine(pbf_file="path/to/your/data.osm")

# Note: PBF files are not directly supported by OSMnx
# Convert PBF to OSM XML format first using tools like osmconvert
```

### Visualization

#### 1. Static Visualization
```python
# Create a static map
router.visualize_route(start_point, end_point, save_path="route.png")
```

#### 2. Interactive Visualization
```python
# Create an interactive HTML map
router.visualize_route_interactive(start_point, end_point)
# Opens route_visualization.html in your browser
```

#### 3. Route Statistics
```python
# Create detailed route analysis
router.plot_route_stats(start_point, end_point)
```

### Complete Example

```python
from router import OSMRoutingEngine

def main():
    # Initialize router
    router = OSMRoutingEngine(place_name="San Francisco, California")
    
    # Print graph information
    print("Graph info:", router.get_graph_info())
    
    # Define route points
    start_point = (-122.4194, 37.7749)  # Union Square
    end_point = (-122.4313, 37.8051)    # Fisherman's Wharf
    
    # Calculate route
    path_coords, distance = router.shortest_path(start_point, end_point)
    
    print(f"Route distance: {distance:.2f} meters")
    
    # Create visualizations
    router.visualize_route(start_point, end_point, save_path="route_static.png")
    router.visualize_route_interactive(start_point, end_point)
    router.plot_route_stats(start_point, end_point)

if __name__ == "__main__":
    main()
```

## 🧪 Testing

### Run All Tests
```bash
python test_router.py
```

### Run with Pytest
```bash
pytest test_router.py -v
```

### Run with Coverage
```bash
pytest test_router.py --cov=router_osmnx --cov-report=html
```

### Test Coverage
The test suite includes:
- Unit tests for all methods
- Integration tests for real-world scenarios
- Edge case testing
- Error handling validation
- Mock testing for external dependencies

## �� API Reference

### OSMRoutingEngine Class

#### Constructor
```python
OSMRoutingEngine(pbf_file=None, place_name=None)
```
- `pbf_file`: Path to OSM XML file (optional)
- `place_name`: Name of place to download (optional)

#### Methods

##### `shortest_path(start_point, end_point)`
Find the shortest path between two coordinates.
- `start_point`: Tuple of (longitude, latitude)
- `end_point`: Tuple of (longitude, latitude)
- Returns: Tuple of (path_coordinates, distance_in_meters)

##### `visualize_route(start_point, end_point, save_path=None)`
Create a static visualization of the route.
- `save_path`: Optional path to save the image

##### `visualize_route_interactive(start_point, end_point)`
Create an interactive HTML map.
- Saves `route_visualization.html` in the current directory

##### `plot_route_stats(start_point, end_point)`
Create detailed route statistics and analysis.

##### `get_graph_info()`
Get information about the loaded graph.
- Returns: Dictionary with graph statistics

##### `validate_coordinates(lon, lat)`
Validate coordinate values.
- Returns: Boolean indicating if coordinates are valid

##### `haversine_distance(lat1, lon1, lat2, lon2)`
Calculate haversine distance between two points.
- Returns: Distance in meters

## 🔧 Configuration

### OSMnx Settings
The engine automatically configures OSMnx settings for optimal performance:
- Adds 'oneway' to useful tags
- Configures network type for driving
- Handles coordinate reference systems

### Error Handling
The engine includes comprehensive error handling for:
- Invalid coordinates
- Missing files
- Network errors
- Empty graphs
- Projection issues

## 🚨 Limitations

1. **PBF Files**: OSMnx doesn't directly support PBF files. Convert to OSM XML format first.
2. **Internet Connection**: Place-based initialization requires internet access.
3. **Memory Usage**: Large areas may require significant memory.
4. **Coordinate System**: Automatic projection may not work for all areas.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone the repository
git clone https://github.com/Mouhamedtec/SPL-Router.git
SPL-Router

# Install development dependencies
pip install -r requirements_test.txt

# Run tests
python test_router.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OSMnx](https://github.com/gboeing/osmnx) - The underlying library for OSM data processing
- [NetworkX](https://networkx.org/) - Graph algorithms and data structures
- [OpenStreetMap](https://www.openstreetmap.org/) - The data source
- [Folium](https://python-visualization.github.io/folium/) - Interactive map visualization

## 📞 Support

If you encounter any issues or have questions:
1. Create a new issue with detailed information
2. Include error messages and system information

---

**Note**: This routing engine is designed for educational and research purposes. For production use, consider additional optimizations and error handling based on your specific requirements.