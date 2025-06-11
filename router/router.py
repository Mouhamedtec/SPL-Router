import osmnx as ox
import networkx as nx
import pandas as pd
import os
import time
from typing import List, Tuple, Optional
from pathlib import Path

class SPLRouterEngine:
    def __init__(self, osm_xml_file: Optional[str] = None, place_name: Optional[str] = None):
        """
        Initialize routing engine with either OSM XML file or place name.
        
        Args:
            osm_xml_file: Path to OSM XML file (optional)
            place_name: Name of place to download (optional)
        """
        self.graph = None

        # Validate input parameters
        if not osm_xml_file and not place_name:
            raise ValueError("Must provide either OSM XML file or place name")
        
        if osm_xml_file and not os.path.exists(osm_xml_file):
            raise FileNotFoundError(f"OSM XML file not found: {osm_xml_file}")

        self.osm_xml_file = osm_xml_file
        self.place_name = place_name
            
        self.load_graph()
    
    def load_graph(self):
        """Load the graph from OSM XML file or OSM download."""
        print("Loading graph...")
        start_time = time.time()
        
        try:
            if self.osm_xml_file:
                # Load from OSM XML file
                ox.settings.useful_tags_way = ox.settings.useful_tags_way + ['oneway']
                self.graph = ox.graph_from_xml(self.osm_xml_file)
            elif self.place_name:
                # Download graph from place name
                self.graph = ox.graph_from_place(self.place_name, network_type='drive')
            else:
                raise ValueError("Must provide either OSM XML file or place name")
            
            # Ensure graph is properly loaded and has data
            if self.graph is None or len(self.graph.nodes()) == 0:
                raise ValueError("Failed to load graph or graph is empty")
            
            # Add edge speeds and travel times
            try:
                self.graph = ox.add_edge_speeds(self.graph)
                self.graph = ox.add_edge_travel_times(self.graph)
            except Exception as e:
                print(f"Warning: Could not add edge speeds/times: {e}")
            
            # Handle projection more carefully
            try:
                # Check if graph is already projected
                if not ox.projection.is_projected(self.graph):
                    print("Projecting graph to UTM coordinate system...")
                    # Project to UTM zone based on graph center
                    self.graph = ox.projection.project_graph(self.graph, to_crs='EPSG:3857')
            except Exception as e:
                print(f"Warning: Could not project graph: {e}")
                print("Continuing with unprojected graph...")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load graph: {str(e)}")
        
        print(f"Graph loaded in {time.time() - start_time:.2f} seconds")
        print(f"Nodes: {len(self.graph.nodes())}")
        print(f"Edges: {len(self.graph.edges())}")
        
        # Print graph CRS info
        try:
            crs = self.graph.graph.get('crs', 'Unknown')
            print(f"Graph CRS: {crs}")
        except:
            print("Could not determine graph CRS")
    
    def validate_coordinates(self, lon: float, lat: float) -> bool:
        """Validate coordinate values."""
        return -180 <= lon <= 180 and -90 <= lat <= 90
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in meters."""
        import math
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Earth radius in meters
        return c * r

    def shortest_path(self, start_point: Tuple[float, float], 
                    end_point: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], float]:
        """
        Find shortest path between two points.
        
        Args:
            start_point: (lon, lat) tuple
            end_point: (lon, lat) tuple
            
        Returns:
            Tuple of (path_coordinates, distance_in_meters)
        """
        # Validate input coordinates
        start_lon, start_lat = start_point
        end_lon, end_lat = end_point
        
        if not self.validate_coordinates(start_lon, start_lat):
            raise ValueError("Invalid start coordinates: longitude must be between -180 and 180, latitude between -90 and 90")
        
        if not self.validate_coordinates(end_lon, end_lat):
            raise ValueError("Invalid end coordinates: longitude must be between -180 and 180, latitude between -90 and 90")
        
        # Check if graph is loaded
        if self.graph is None or len(self.graph.nodes()) == 0:
            raise RuntimeError("Graph is not loaded or empty")
        
        try:
            # Find nearest nodes to the input points
            start_node = ox.distance.nearest_nodes(self.graph, start_lon, start_lat)
            end_node = ox.distance.nearest_nodes(self.graph, end_lon, end_lat)
            
            # Validate that we found valid nodes
            if start_node not in self.graph.nodes():
                raise ValueError(f"Start node {start_node} not found in graph")
            if end_node not in self.graph.nodes():
                raise ValueError(f"End node {end_node} not found in graph")
            
            # Handle case where start and end are the same
            if start_node == end_node:
                node_data = self.graph.nodes[start_node]
                return [(node_data.get('x', start_lon), node_data.get('y', start_lat))], 0.0
            
            # Check if graph is projected for accurate distance calculations
            is_projected = False
            try:
                is_projected = ox.projection.is_projected(self.graph)
            except:
                pass
            
            # Calculate shortest path
            if is_projected:
                # Use length attribute for projected graphs
                route = nx.shortest_path(
                    self.graph, 
                    start_node, 
                    end_node, 
                    weight='length'
                )
            else:
                # Use haversine distance for unprojected graphs
                def haversine_weight(u, v, d):
                    u_data = self.graph.nodes[u]
                    v_data = self.graph.nodes[v]
                    return self.haversine_distance(
                        u_data.get('y', 0), u_data.get('x', 0),
                        v_data.get('y', 0), v_data.get('x', 0)
                    )
                
                route = nx.shortest_path(
                    self.graph, 
                    start_node, 
                    end_node, 
                    weight=haversine_weight
                )
            
            # Get route coordinates and distance
            route_coords = []
            for n in route:
                node_data = self.graph.nodes[n]
                x = node_data.get('x', 0)
                y = node_data.get('y', 0)
                route_coords.append((x, y))
            
            # Calculate total distance
            distance = 0.0
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                if self.graph.has_edge(u, v):
                    if is_projected:
                        edge_data = self.graph[u][v]
                        distance += edge_data.get('length', 0.0)
                    else:
                        # Use haversine distance for unprojected graphs
                        u_data = self.graph.nodes[u]
                        v_data = self.graph.nodes[v]
                        distance += self.haversine_distance(
                            u_data.get('y', 0), u_data.get('x', 0),
                            v_data.get('y', 0), v_data.get('x', 0)
                        )
            
            return route_coords, distance
            
        except nx.NetworkXNoPath:
            raise ValueError("No path found between the specified points")
        except Exception as e:
            raise RuntimeError(f"Error calculating shortest path: {str(e)}")
    
    def get_graph_info(self) -> dict:
        """Get information about the loaded graph."""
        if self.graph is None:
            return {"error": "Graph not loaded"}
        
        info = {
            "nodes": len(self.graph.nodes()),
            "edges": len(self.graph.edges()),
        }
        
        # Safely check projection status
        try:
            info["is_projected"] = ox.projection.is_projected(self.graph)
        except:
            info["is_projected"] = "Unknown"
        
        # Safely get CRS info
        try:
            info["crs"] = self.graph.graph.get('crs', 'Unknown')
        except:
            info["crs"] = "Unknown"
        
        return info

    def visualize_route(self, start_point: Tuple[float, float], 
                       end_point: Tuple[float, float], 
                       save_path: Optional[str] = None) -> None:
        """
        Visualize the route between two points.
        
        Args:
            start_point: (lon, lat) tuple for start
            end_point: (lon, lat) tuple for end
            save_path: Optional path to save the plot as image
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get the route
            route_coords, distance = self.shortest_path(start_point, end_point)
            
            # Extract node IDs for the route
            start_lon, start_lat = start_point
            end_lon, end_lat = end_point
            
            start_node = ox.distance.nearest_nodes(self.graph, start_lon, start_lat)
            end_node = ox.distance.nearest_nodes(self.graph, end_lon, end_lat)
            
            # Get the route as node IDs
            is_projected = False
            try:
                is_projected = ox.projection.is_projected(self.graph)
            except:
                pass
            
            if is_projected:
                route_nodes = nx.shortest_path(self.graph, start_node, end_node, weight='length')
            else:
                def haversine_weight(u, v, d):
                    u_data = self.graph.nodes[u]
                    v_data = self.graph.nodes[v]
                    return self.haversine_distance(
                        u_data.get('y', 0), u_data.get('x', 0),
                        v_data.get('y', 0), v_data.get('x', 0)
                    )
                route_nodes = nx.shortest_path(self.graph, start_node, end_node, weight=haversine_weight)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot the route
            ox.plot_graph_route(self.graph, route_nodes, ax=ax, 
                              route_color='red', route_linewidth=3, 
                              route_alpha=0.8, show=False)
            
            # Add start and end markers
            ax.scatter(start_lon, start_lat, c='green', s=100, marker='o', 
                      label='Start', zorder=5, edgecolors='black', linewidth=2)
            ax.scatter(end_lon, end_lat, c='red', s=100, marker='s', 
                      label='End', zorder=5, edgecolors='black', linewidth=2)
            
            # Add title and legend
            ax.set_title(f'Route from {start_point} to {end_point}\nDistance: {distance:.2f} meters')
            ax.legend()
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Route visualization saved to: {save_path}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib is required for visualization. Install with: pip install matplotlib")
        except Exception as e:
            print(f"Error visualizing route: {str(e)}")

    def visualize_route_interactive(self, start_point: Tuple[float, float], 
                                  end_point: Tuple[float, float]) -> None:
        """
        Create an interactive visualization using folium.
        
        Args:
            start_point: (lon, lat) tuple for start
            end_point: (lon, lat) tuple for end
        """
        try:
            import folium
            
            # Get the route
            route_coords, distance = self.shortest_path(start_point, end_point)
            
            # Calculate center point for the map
            center_lat = (start_point[1] + end_point[1]) / 2
            center_lon = (start_point[0] + end_point[0]) / 2
            
            # Create the map
            m = folium.Map(location=[center_lat, center_lon], 
                          zoom_start=13, 
                          tiles='OpenStreetMap')
            
            # Add start marker
            folium.Marker(
                location=[start_point[1], start_point[0]],
                popup=f'Start<br>Distance: {distance:.2f}m',
                icon=folium.Icon(color='green', icon='info-sign')
            ).add_to(m)
            
            # Add end marker
            folium.Marker(
                location=[end_point[1], end_point[0]],
                popup=f'End<br>Distance: {distance:.2f}m',
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
            # Add route line
            route_locations = [[coord[1], coord[0]] for coord in route_coords]
            folium.PolyLine(
                locations=route_locations,
                color='red',
                weight=4,
                opacity=0.8,
                popup=f'Route<br>Distance: {distance:.2f}m'
            ).add_to(m)
            
            # Save the map
            map_path = 'route_visualization.html'
            m.save(map_path)
            print(f"Interactive route visualization saved to: {map_path}")
            print("Open the HTML file in your web browser to view the interactive map.")
            
        except ImportError:
            print("Folium is required for interactive visualization. Install with: pip install folium")
        except Exception as e:
            print(f"Error creating interactive visualization: {str(e)}")

    def plot_route_stats(self, start_point: Tuple[float, float], 
                        end_point: Tuple[float, float]) -> None:
        """
        Plot route statistics and analysis.
        
        Args:
            start_point: (lon, lat) tuple for start
            end_point: (lon, lat) tuple for end
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Get the route
            route_coords, distance = self.shortest_path(start_point, end_point)
            
            # Calculate additional statistics
            distances = []
            for i in range(len(route_coords) - 1):
                dist = self.haversine_distance(
                    route_coords[i][1], route_coords[i][0],
                    route_coords[i+1][1], route_coords[i+1][0]
                )
                distances.append(dist)
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Route overview
            ax1.plot([coord[0] for coord in route_coords], 
                    [coord[1] for coord in route_coords], 
                    'r-', linewidth=2, label='Route')
            ax1.scatter(start_point[0], start_point[1], c='green', s=100, 
                       marker='o', label='Start', zorder=5)
            ax1.scatter(end_point[0], end_point[1], c='red', s=100, 
                       marker='s', label='End', zorder=5)
            ax1.set_title('Route Overview')
            ax1.legend()
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            
            # Plot 2: Distance histogram
            if distances:
                ax2.hist(distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax2.set_title('Segment Distance Distribution')
                ax2.set_xlabel('Distance (meters)')
                ax2.set_ylabel('Frequency')
            
            # Plot 3: Cumulative distance
            cumulative_dist = np.cumsum([0] + distances)
            ax3.plot(range(len(cumulative_dist)), cumulative_dist, 'b-', linewidth=2)
            ax3.set_title('Cumulative Distance')
            ax3.set_xlabel('Route Segment')
            ax3.set_ylabel('Cumulative Distance (meters)')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Route statistics
            ax4.axis('off')
            stats_text = f"""
Route Statistics:
• Total Distance: {distance:.2f} meters
• Number of Segments: {len(route_coords) - 1}
• Average Segment Length: {np.mean(distances):.2f} meters
• Max Segment Length: {np.max(distances):.2f} meters
• Min Segment Length: {np.min(distances):.2f} meters
• Start: {start_point}
• End: {end_point}
            """
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib and NumPy are required for statistics plotting. Install with: pip install matplotlib numpy")
        except Exception as e:
            print(f"Error plotting route statistics: {str(e)}")
