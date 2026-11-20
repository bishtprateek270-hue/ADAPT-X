"""
ADAPT-X: Urban Obstacle Map, Artificial Potential Field (APF) & Canyon Corridor Router
Provides geometric collision bounds, repulsive potential fields, and collision-free
avenue pathfinding for city drone flight through urban skyscraper canyons.
"""

import numpy as np


# Bounding boxes for 3D City Buildings [xmin, xmax, ymin, ymax, zmax]
BUILDING_BOXES = [
    # Tower NE: The Grand Horizon Skyscraper (x=16, y=16)
    {"name": "HORIZON TOWER", "box": (11.0, 21.0, 11.0, 21.0, 27.0)},
    # Tower NW: Old Metropolitan Brick Lofts (x=-16, y=16)
    {"name": "METRO LOFTS", "box": (-21.0, -11.0, 11.0, 21.0, 15.0)},
    # Tower SW: Apex Corporate Financial Center (x=-16, y=-16)
    {"name": "APEX CORP", "box": (-21.0, -11.0, -21.0, -11.0, 20.0)},
    # Tower SE: Vanguard Residences (x=16, y=-16)
    {"name": "VANGUARD", "box": (11.0, 21.0, -21.0, -11.0, 19.0)},
    # North Avenue Commercial Mid-Rise (pos=0, 30)
    {"name": "BLOCK NORTH", "box": (-9.0, 9.0, 25.5, 34.5, 14.0)},
    # South Avenue High-Tech Campus (pos=0, -30)
    {"name": "BLOCK SOUTH", "box": (-8.5, 8.5, -34.5, -25.5, 16.0)},
    # East Boulevard Grand Hotel (pos=30, 0)
    {"name": "BLOCK EAST", "box": (25.5, 34.5, -9.0, 9.0, 14.0)},
    # West Boulevard Financial Center (pos=-30, 0)
    {"name": "BLOCK WEST", "box": (-34.5, -25.5, -8.5, 8.5, 15.0)},
]

# Cylindrical street-level obstacles [x, y, radius, height]
CYLINDER_OBSTACLES = [
    # Traffic Light Gantry (x=3.6 to 5.8, y=7.0)
    (4.7, 7.0, 1.4, 4.9),
    # Street Lamps (h=3.8m)
    (6.8, 6.8, 0.6, 3.9),
    (-6.8, 6.8, 0.6, 3.9),
    (6.8, -6.8, 0.6, 3.9),
    (-6.8, -6.8, 0.6, 3.9),
    # Urban Trees (h=3.7m)
    (8.5, 8.5, 1.4, 3.7),
    (-8.5, 8.5, 1.4, 3.7),
    (8.5, -8.5, 1.4, 3.7),
    (-8.5, -8.5, 1.4, 3.7),
    (8.5, 19.0, 1.3, 3.7),
    (-8.5, 19.0, 1.3, 3.7),
    (19.0, 8.5, 1.3, 3.7),
    (-19.0, 8.5, 1.3, 3.7),
]


class CityObstacleMap:
    """
    Real-time geometric collision evaluator and Artificial Potential Field generator.
    """
    def __init__(self, drone_radius=0.45, safe_buffer=1.8):
        self.drone_radius = drone_radius
        self.safe_buffer = safe_buffer
        self.d_safe = drone_radius + safe_buffer  # ~2.25m clearance boundary

    def is_inside_building(self, x, y, z=4.5, margin=0.6):
        """Checks if 3D coordinate (x, y, z) penetrates a building."""
        for b in BUILDING_BOXES:
            xmin, xmax, ymin, ymax, zmax = b["box"]
            if z <= zmax and (xmin - margin <= x <= xmax + margin) and (ymin - margin <= y <= ymax + margin):
                return True
        return False

    def snap_to_safe_corridor(self, x, y, z=5.2):
        """Snaps a point inside an obstacle to the nearest safe street avenue."""
        clamped_x = float(np.clip(x, -19.5, 19.5))
        clamped_y = float(np.clip(y, -19.5, 19.5))

        if not self.is_inside_building(clamped_x, clamped_y, z, margin=0.6):
            return np.array([clamped_x, clamped_y, z])

        # Find closest safe avenue corridor
        # Avenues run at x=0 (N-S), y=0 (E-W), and outer boundaries
        safe_candidates = [
            np.array([0.0, clamped_y, z]),             # North-South Avenue
            np.array([clamped_x, 0.0, z]),             # East-West Boulevard
            np.array([clamped_x, np.sign(clamped_y) * 6.5, z]),  # Plaza avenue edge
            np.array([np.sign(clamped_x) * 6.5, clamped_y, z]),  # Plaza avenue edge
            np.array([np.sign(clamped_x) * 23.0, clamped_y, z]), # Outer street
            np.array([clamped_x, np.sign(clamped_y) * 23.0, z]), # Outer street
        ]

        valid = [pt for pt in safe_candidates if not self.is_inside_building(pt[0], pt[1], z, margin=0.3)]
        if valid:
            dists = [np.hypot(pt[0] - clamped_x, pt[1] - clamped_y) for pt in valid]
            return valid[int(np.argmin(dists))]
        return np.array([0.0, 0.0, z])

    def line_intersects_building(self, p1, p2, z=5.2, margin=0.9):
        """Liang-Barsky test: checks if segment p1->p2 intersects any building box."""
        x1, y1 = p1[:2]
        x2, y2 = p2[:2]
        dx = x2 - x1
        dy = y2 - y1

        for b in BUILDING_BOXES:
            xmin, xmax, ymin, ymax, zmax = b["box"]
            if z > zmax:
                continue

            bx_min = xmin - margin
            bx_max = xmax + margin
            by_min = ymin - margin
            by_max = ymax + margin

            p = [-dx, dx, -dy, dy]
            q = [x1 - bx_min, bx_max - x1, y1 - by_min, by_max - y1]
            u1, u2 = 0.0, 1.0
            intersect = True

            for i in range(4):
                if abs(p[i]) < 1e-9:
                    if q[i] < 0:
                        intersect = False
                        break
                else:
                    t = q[i] / p[i]
                    if p[i] < 0:
                        if t > u2:
                            intersect = False
                            break
                        if t > u1:
                            u1 = t
                    else:
                        if t < u1:
                            intersect = False
                            break
                        if t < u2:
                            u2 = t

            if intersect and u1 <= u2:
                return True
        return False

    def plan_corridor_route(self, start, goal, z=5.2):
        """
        Plans a collision-free waypoint sequence using the city's open boulevards
        and avenue corridors around the skyscrapers.
        """
        # Snap start and goal outside any building
        s = np.array([float(start[0]), float(start[1]), z])
        g = self.snap_to_safe_corridor(float(goal[0]), float(goal[1]), z)

        # 1. Direct path check: If line of sight is clear, fly straight
        if not self.line_intersects_building(s, g, z):
            return [g]

        # 2. Key City Avenue Waypoint Nodes (Boulevard corridors)
        # Main thoroughfares: N-S Avenue (x=0), E-W Boulevard (y=0)
        nav_nodes = {
            "center": np.array([0.0, 0.0, z]),
            "north_plaza": np.array([0.0, 6.5, z]),
            "north_mid": np.array([0.0, 16.0, z]),
            "north_far": np.array([0.0, 23.5, z]),
            "south_plaza": np.array([0.0, -6.5, z]),
            "south_mid": np.array([0.0, -16.0, z]),
            "south_far": np.array([0.0, -23.5, z]),
            "east_plaza": np.array([6.5, 0.0, z]),
            "east_mid": np.array([16.0, 0.0, z]),
            "east_far": np.array([23.5, 0.0, z]),
            "west_plaza": np.array([-6.5, 0.0, z]),
            "west_mid": np.array([-16.0, 0.0, z]),
            "west_far": np.array([-23.5, 0.0, z]),
        }

        # 3. Two-segment routing via candidate avenue hubs
        best_path = None
        min_total_dist = float("inf")

        for _name, hub in nav_nodes.items():
            if not self.line_intersects_building(s, hub, z) and not self.line_intersects_building(hub, g, z):
                dist = float(np.hypot(s[0] - hub[0], s[1] - hub[1]) + np.hypot(hub[0] - g[0], hub[1] - g[1]))
                if dist < min_total_dist:
                    min_total_dist = dist
                    best_path = [hub, g]

        if best_path is not None:
            return best_path

        # 4. Three-segment routing via Manhattan avenue network
        # E.g. [start] -> [closest axis point] -> [intersection] -> [goal axis point] -> [goal]
        hub_s = np.array([s[0], 0.0, z]) if abs(s[1]) < abs(s[0]) else np.array([0.0, s[1], z])
        hub_g = np.array([g[0], 0.0, z]) if abs(g[1]) < abs(g[0]) else np.array([0.0, g[1], z])

        path = []
        curr = s
        for h in [hub_s, nav_nodes["center"], hub_g]:
            if np.hypot(curr[0] - h[0], curr[1] - h[1]) > 0.8:
                if not self.line_intersects_building(curr, h, z):
                    path.append(h)
                    curr = h
        path.append(g)
        return path

    def compute_apf_force(self, pos, vel):
        """
        Computes Artificial Potential Field repulsive force and vortex circulation
        force away from all buildings, trees, and street furniture.
        
        Returns:
            f_rep (np.ndarray): 3D repulsive force vector [fx, fy, fz]
            min_dist (float): Distance to closest obstacle in meters
        """
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        vx, vy = float(vel[0]), float(vel[1])
        v_speed = float(np.hypot(vx, vy))

        f_total = np.zeros(3, dtype=np.float64)
        min_dist = 999.0

        # --- 1. Skyscraper & Avenue Infill Bounding Boxes ---
        for b in BUILDING_BOXES:
            xmin, xmax, ymin, ymax, zmax = b["box"]
            if pz > zmax + 0.5:
                continue

            cx = float(np.clip(px, xmin, xmax))
            cy = float(np.clip(py, ymin, ymax))
            dx = px - cx
            dy = py - cy
            dist_2d = float(np.hypot(dx, dy))
            min_dist = min(min_dist, dist_2d)

            # Check if within safety activation zone (2.5m from facade)
            if dist_2d < self.d_safe:
                if dist_2d < 1e-4:
                    # Inside box edge fallback
                    normal = np.array([px - (xmin + xmax)/2.0, py - (ymin + ymax)/2.0])
                    norm_len = float(np.hypot(normal[0], normal[1]))
                    normal = normal / (norm_len + 1e-5)
                    eff_dist = 0.05
                else:
                    normal = np.array([dx, dy]) / dist_2d
                    eff_dist = max(0.05, dist_2d)

                # Strong repulsive force pushing away from the building facade
                # Repulsion magnitude scales inversely with clearance
                f_mag = float(np.clip(16.0 * (1.0 / eff_dist - 1.0 / self.d_safe), 0.0, 32.0))
                f_rep_2d = f_mag * normal

                # Vortex circulation / tangential sliding force around building corners
                # Only active when drone is moving past obstacle (v_speed >= 0.35 m/s)
                # Suppressed when stationary or hovering to guarantee rock-solid stationary position lock
                if v_speed >= 0.35:
                    tangent = np.array([-normal[1], normal[0]])
                    dot_v = (vx * tangent[0] + vy * tangent[1]) / v_speed
                    sign_tan = 1.0 if dot_v >= 0 else -1.0
                    f_tan_2d = min(3.0, 3.0 * v_speed) * sign_tan * tangent
                else:
                    f_tan_2d = np.zeros(2)

                f_total[0] += f_rep_2d[0] + f_tan_2d[0]
                f_total[1] += f_rep_2d[1] + f_tan_2d[1]

        # --- 2. Cylindrical Street Furniture & Trees (Lamps, Gantries, Foliage) ---
        for ox, oy, r, h in CYLINDER_OBSTACLES:
            # Check elevation clearance first: ignore ground obstacles when flying overhead
            if pz > (h + 0.3):
                continue

            dx = px - ox
            dy = py - oy
            d_center = float(np.hypot(dx, dy))
            d_surf = max(0.01, d_center - r)
            min_dist = min(min_dist, d_surf)

            # Check if drone is at or below obstacle height and within buffer
            if d_surf < 1.6:
                normal = np.array([dx, dy]) / (d_center + 1e-5)
                f_mag = float(np.clip(12.0 * (1.0 / d_surf - 1.0 / 1.6), 0.0, 24.0))
                f_total[0] += f_mag * normal[0]
                f_total[1] += f_mag * normal[1]

                # If below obstacle height, add gentle vertical clearance assist
                if pz < h:
                    f_total[2] += float(np.clip(6.0 * (h + 0.2 - pz), 0.0, 8.0))

        # --- 3. Outer City Geofence Containment Field (|x|, |y| < 21m) ---
        for axis_idx, coord in enumerate([px, py]):
            if abs(coord) > 19.5:
                delta = abs(coord) - 19.5
                f_total[axis_idx] -= np.sign(coord) * float(np.clip(18.0 * delta, 0.0, 35.0))

        return f_total, min_dist

    def compute_lidar_distances(self, pos, num_beams=8, max_range=12.0):
        """
        Computes 360-degree horizontal LiDAR range measurements for RL state observation.
        Beams are evenly distributed in radians: [0, 2pi / num_beams, ...].
        """
        px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
        angles = np.linspace(0.0, 2.0 * np.pi, num_beams, endpoint=False)
        ranges = np.full(num_beams, max_range, dtype=np.float32)

        for b_idx, theta in enumerate(angles):
            ray_dx = float(np.cos(theta))
            ray_dy = float(np.sin(theta))
            ray_end = np.array([px + ray_dx * max_range, py + ray_dy * max_range, pz])

            # Check ray intersection with all buildings
            for b in BUILDING_BOXES:
                xmin, xmax, ymin, ymax, zmax = b["box"]
                if pz > zmax:
                    continue

                # Ray-AABB intersection in 2D
                tmin = 0.0
                tmax = max_range

                for axis, ray_d, p_coord, b_min, b_max in [
                    (0, ray_dx, px, xmin, xmax),
                    (1, ray_dy, py, ymin, ymax)
                ]:
                    if abs(ray_d) < 1e-8:
                        if p_coord < b_min or p_coord > b_max:
                            tmin = float("inf")
                            break
                    else:
                        t1 = (b_min - p_coord) / ray_d
                        t2 = (b_max - p_coord) / ray_d
                        if t1 > t2:
                            t1, t2 = t2, t1
                        tmin = max(tmin, t1)
                        tmax = min(tmax, t2)
                        if tmin > tmax:
                            break

                if tmin <= tmax and tmin >= 0.0:
                    ranges[b_idx] = min(ranges[b_idx], float(tmin))

        return ranges

