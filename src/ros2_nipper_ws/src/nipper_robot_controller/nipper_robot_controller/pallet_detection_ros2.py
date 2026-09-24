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

# --- Utility functions for cluster centroid logic ---
def are_parallel(c1, c2, tol_dist=0.06, target_dist=0.8):  # Euro pallet width = 0.8m
    dist = np.linalg.norm(c1 - c2)
    return abs(dist - target_dist) < tol_dist

def are_colinear(c1, c2, c3, tol_angle=0.15):   # radians
    v1 = c2 - c1
    v2 = c3 - c1
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return False
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Clamp due to rounding
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return angle < tol_angle or abs(angle - np.pi) < tol_angle

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
        centroids = [np.mean(cluster, axis=0) for cluster in clusters]

        # Save parallel pairs and their directions
        parallel_pairs = []
        parallel_dirs = []

        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                if are_parallel(centroids[i], centroids[j]):
                    A = centroids[i]
                    B = centroids[j]
                    direction = B - A
                    norm = np.linalg.norm(direction)
                    if norm == 0:
                        continue
                    direction_unit = direction / norm
                    blocked = False
                    for k in range(len(centroids)):
                        if k == i or k == j:
                            continue
                        C = centroids[k]
                        proj = np.dot(C - A, direction_unit) / norm
                        # If projection is beyond B (>1), C is blocking/behind extension
                        if proj > 1.0:
                            blocked = True
                            break
                    if not blocked:
                        line_marker = Marker()
                        line_marker.header.frame_id = msg.header.frame_id
                        line_marker.header.stamp = msg.header.stamp
                        line_marker.ns = "pallet_edges"
                        line_marker.id = 500 + i * len(centroids) + j
                        line_marker.type = Marker.LINE_LIST
                        line_marker.action = Marker.ADD
                        line_marker.scale.x = 0.04
                        line_marker.color.r = 1.0
                        line_marker.color.g = 0.0
                        line_marker.color.b = 0.0
                        line_marker.color.a = 1.0
                        line_marker.points = [
                            Point(x=float(A[0]), y=float(A[1]), z=0.05),
                            Point(x=float(B[0]), y=float(B[1]), z=0.05)
                        ]
                        markers.markers.append(line_marker)
                        parallel_pairs.append((i, j))
                        parallel_dirs.append(direction_unit)

        used_parallel = set([idx for pair in parallel_pairs for idx in pair])

        # Detect colinear (slat) lines and mark involved indices
        used_colinear = set()
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                for k in range(j+1, len(centroids)):
                    if are_colinear(centroids[i], centroids[j], centroids[k]):
                        used_colinear.update([i, j, k])
                        # Draw line between furthest two
                        dists = [
                            (np.linalg.norm(centroids[i] - centroids[j]), (i, j)),
                            (np.linalg.norm(centroids[i] - centroids[k]), (i, k)),
                            (np.linalg.norm(centroids[j] - centroids[k]), (j, k))
                        ]
                        dists.sort(reverse=True)
                        idx1, idx2 = dists[0][1]
                        slat_marker = Marker()
                        slat_marker.header.frame_id = msg.header.frame_id
                        slat_marker.header.stamp = msg.header.stamp
                        slat_marker.ns = "pallet_slats"
                        slat_marker.id = 1000 + i * len(centroids) + j * len(centroids) + k
                        slat_marker.type = Marker.LINE_LIST
                        slat_marker.action = Marker.ADD
                        slat_marker.scale.x = 0.02
                        slat_marker.color.r = 0.3
                        slat_marker.color.g = 0.2
                        slat_marker.color.b = 0.6
                        slat_marker.color.a = 1.0
                        slat_marker.points = [
                            Point(x=float(centroids[idx1][0]), y=float(centroids[idx1][1]), z=0.05),
                            Point(x=float(centroids[idx2][0]), y=float(centroids[idx2][1]), z=0.05)
                        ]
                        markers.markers.append(slat_marker)

        # If two or more unique parallel directions exist, draw a crossbar for unused centroids
        if len(parallel_dirs) >= 1:
            ref_dir = parallel_dirs[0]
            perp_dir = np.array([-ref_dir[1], ref_dir[0]])  # 90 degree rotation
            # Estimate line length: use max distance among parallel pairs (or fixed length if desired)
            lengths = [np.linalg.norm(centroids[i] - centroids[j]) for (i,j) in parallel_pairs]
            line_length = max(lengths) if lengths else 1.2  # default to euro-pallet length if none
            for idx in range(len(centroids)):
                if idx not in used_parallel and idx not in used_colinear:
                    C = centroids[idx]
                    start = C - perp_dir * (line_length / 2)
                    end = C + perp_dir * (line_length / 2)
                    cross_marker = Marker()
                    cross_marker.header.frame_id = msg.header.frame_id
                    cross_marker.header.stamp = msg.header.stamp
                    cross_marker.ns = "special_pallet_cross"
                    cross_marker.id = 2000 + idx
                    cross_marker.type = Marker.LINE_LIST
                    cross_marker.action = Marker.ADD
                    cross_marker.scale.x = 0.025
                    cross_marker.color.r = 0.0
                    cross_marker.color.g = 1.0
                    cross_marker.color.b = 1.0
                    cross_marker.color.a = 1.0
                    cross_marker.points = [
                        Point(x=float(start[0]), y=float(start[1]), z=0.05),
                        Point(x=float(end[0]), y=float(end[1]), z=0.05)
                    ]
                    markers.markers.append(cross_marker)

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