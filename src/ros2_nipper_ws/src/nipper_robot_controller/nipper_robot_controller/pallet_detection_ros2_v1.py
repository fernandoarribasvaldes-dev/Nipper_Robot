#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Point

def laser_scan_to_xy(scan_msg):
    angles = np.arange(scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment)
    ranges = np.array(scan_msg.ranges)
    # Remove inf/nan
    valid = np.isfinite(ranges)
    ranges = ranges[valid]
    angles = angles[valid]
    # Convert to x, y
    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)
    return np.stack((xs, ys), axis=-1)

def preprocess(points, min_range=0.2, max_range=10.0):
    dists = np.linalg.norm(points, axis=1)
    mask = (dists > min_range) & (dists < max_range)
    return points[mask]

def cluster_points(points, eps=0.15, min_samples=8):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique = set(labels) - {-1}
    clusters = [points[labels==l] for l in unique]
    return clusters

def is_pallet_shape(cluster, min_size=(0.08, 0.05), max_size=(0.3, 0.1), aspect_tolerance=0.3):
    if len(cluster) < 4:
        return None
    from scipy.spatial import ConvexHull
    hull = ConvexHull(cluster)
    coords = cluster[hull.vertices]
    centroid = np.mean(coords, axis=0)
    u, s, vh = np.linalg.svd(coords - centroid)
    rot = vh.T
    proj = (coords - centroid) @ rot
    min_x, min_y = np.min(proj, axis=0)
    max_x, max_y = np.max(proj, axis=0)
    w, h = max_x - min_x, max_y - min_y
    # Aspect ratio and size check
    aspect = min(w, h) / max(w, h)
    print(f"Cluster dims: w={w:.2f}, h={h:.2f}, aspect={aspect:.2f}, points={len(cluster)}")
    if ((min_size[0] <= w <= max_size[0] and min_size[1] <= h <= max_size[1]) or
        (min_size[1] <= w <= max_size[1] and min_size[0] <= h <= max_size[0])):
        if aspect > aspect_tolerance:
            yaw = np.arctan2(rot[1, 0], rot[0, 0])
            return centroid, (w, h), yaw
    return None

def get_rectangle_points(centroid, dims, angle):
    w, h = dims
    half_w, half_h = w/2, h/2
    corners = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h],
        [-half_w, -half_h]  # Close the rectangle
    ])
    rot = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    rotated = (rot @ corners.T).T
    translated = rotated + centroid
    points = [Point(x=float(pt[0]), y=float(pt[1]), z=0.05) for pt in translated]
    return points



class PalletDetectionNode(Node):
    def __init__(self):
        super().__init__('pallet_detection_node')
        self.sub = self.create_subscription(
            LaserScan, '/scan_rear', self.lidar_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'pallet_markers', 10)
        self.marker_seq = 0

    def lidar_callback(self, msg):
        points = laser_scan_to_xy(msg)
        print(f"Points from scan: {points.shape[0]}")
        processed = preprocess(points)
        print(f"After preprocess: {processed.shape[0]}")
        clusters = cluster_points(processed)
        print(f"Clusters found: {len(clusters)}")
        markers = MarkerArray()
        id_num = 0
        for idx, cluster in enumerate(clusters):
            print(f"  Cluster {idx}: size={cluster.shape[0]}")
            result = is_pallet_shape(cluster)
            if result is not None:
                centroid, dims, angle = result
                print(f"Pallet detected at {centroid}, size={dims}, angle={angle}")
                marker = Marker()
                marker.header = Header()
                marker.header.frame_id = msg.header.frame_id
                marker.header.stamp = msg.header.stamp
                marker.ns = "pallets"
                marker.id = id_num
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = float(centroid[0])
                marker.pose.position.y = float(centroid[1])
                marker.pose.position.z = 0.05  # Slightly above ground
                # Orientation: yaw
                # from tf_transformations import quaternion_from_euler
                # q = quaternion_from_euler(0, 0, angle)
                r = R.from_euler('z', angle)
                q = r.as_quat()  # Returns [x, y, z, w]
                marker.pose.orientation.x = float(q[0])
                marker.pose.orientation.y = float(q[1])
                marker.pose.orientation.z = float(q[2])
                marker.pose.orientation.w = float(q[3])
                marker.scale.x = float(dims[0])
                marker.scale.y = float(dims[1])
                marker.scale.z = 0.1
                marker.color = ColorRGBA()
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.8
                marker.lifetime.sec = 0
                marker.lifetime.nanosec = 500000000  # Half second
                markers.markers.append(marker)

                # ------------ Pallet outline marker ---------------
                outline_marker = Marker()
                outline_marker.header = marker.header
                outline_marker.ns = "pallet_outline"
                outline_marker.id = id_num + 100  # offset IDs
                outline_marker.type = Marker.LINE_STRIP
                outline_marker.action = Marker.ADD
                outline_marker.scale.x = 0.02
                outline_marker.color.r = 1.0
                outline_marker.color.g = 1.0
                outline_marker.color.b = 0.0
                outline_marker.color.a = 1.0
                outline_marker.lifetime.sec = 0
                outline_marker.lifetime.nanosec = 500000000
                outline_marker.points = get_rectangle_points(centroid, dims, angle)
                markers.markers.append(outline_marker)
                # --------------------------------------------------

            
                id_num += 1
        # Remove leftover stale markers
        for j in range(id_num, 20):
            marker = Marker()
            marker.header = Header()
            marker.header.frame_id = msg.header.frame_id
            marker.id = j
            marker.action = Marker.DELETE
            markers.markers.append(marker)
        self.marker_pub.publish(markers)

def main(args=None):
    rclpy.init(args=args)
    node = PalletDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()