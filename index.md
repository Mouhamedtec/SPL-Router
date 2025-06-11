# SPL-Router Documentation

## Overview
SPL-Router is a robust routing engine built on top of OSMnx for OpenStreetMap data processing and route calculation.

## Installation
```bash
pip install spl-router
```

## Quick Start
```python
from router import SPLRouterEngine

# Initialize with a place name
router = SPLRouterEngine(place_name="San Francisco")

# Or with an OSM XML file
router = SPLRouterEngine(osm_xml_file="map.osm")

# Calculate route
start_point = (-122.4194, 37.7749)  # San Francisco
end_point = (-122.4313, 37.8051)    # Fisherman's Wharf
route, distance = router.shortest_path(start_point, end_point)
```

## Features
- Flexible Data Sources: Support for both place names and OSM XML files
- Multiple Visualization Options
- Comprehensive Route Statistics
- Reverse Geocoding Support