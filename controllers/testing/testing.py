import cv2
import numpy as np
import math
import yaml
import csv
from dt_apriltags import Detector

# Webots Imports
from vehicle import Driver
from controller import Keyboard

# Local Imports
from bev_calibrator import BEVCalibrator

# --- Configuration ---
MAX_SPEED_WEBOTS = 50.0 
CRUISING_SPEED = 100.0
STANLEY_K = 3.0
TAG_MAP_FILE = "output3.yaml" 
FRONT_SLOT_Y = 0.0
FRONT_SLOT_Z = 0.45
FRONT_SLOT_X = 3.85
CAMERA_X = 0.0954
CAMERA_Y = 0.0
CAMERA_Z = 0.169114
POSE_LOOKAHEAD_M = 2.5 # Meters to project ahead for pose estimation

# --- Odometry Configuration ---
WHEEL_RADIUS = 0.40  # Meters (Approx for BMW X5 in Webots)
WHEEL_BASE = 2.995   # Meters (Distance between front and rear axles)
SPEED_NOISE_STD = 0.5 # Standard deviation for speed noise (km/h)
STEER_NOISE_STD = 0.005 # Standard deviation for steering noise (radians)

# --- Helper Functions ---
def _contour_center(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0: return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

class DataLogger:
    def __init__(self, filename="result.csv"):
        self.filename = filename
        self.file = open(self.filename, mode='w', newline='', buffering=1)
        self.writer = csv.writer(self.file)
        
        # Updated Header: Renamed pos_ -> gps_ and added vis_ columns + theta_r
        header = [
            "timestamp", 
            "gps_x", "gps_y", "gps_z",       # Car GPS Position (Renamed)
            "track_x", "track_y", "track_z", # Calculated Track Center Position
            "roll", "pitch", "yaw",          # Car Orientation (IMU)
            "cross_track_error", "heading_error", 
            "speed_cmd", "steering_cmd", "has_lock",
            "odom_x", "odom_y", "odom_yaw",  # Odometry
            "vis_x", "vis_y", "vis_z",       # Vision POSE POS
            "vis_roll", "vis_pitch", "vis_yaw", # Vision POSE ROT
            "theta_r"                        # Path heading
        ]
        self.writer.writerow(header)
        print(f"Data logger initialized. Writing to: {self.filename}")

    def log(self, timestamp, pos, track_pos, rot, cte, he, speed, steer, has_lock, odom, vision_pose, theta_r):
        px, py, pz = pos if pos is not None else (None, None, None)
        tx, ty, tz = track_pos if track_pos is not None else (None, None)
        roll, pitch, yaw = rot if rot is not None else (None, None, None)
        
        # Unpack Odometry
        ox, oy, oyaw = odom

        # Unpack Vision Pose
        # vision_pose is expected to be ((x,y,z), (r,p,y)) or None
        if vision_pose and vision_pose[0] is not None:
            v_trans, v_rot = vision_pose
            vx, vy, vz = v_trans[0], v_trans[1], v_trans[2]
            vr, vp, vy_ang = v_rot[0], v_rot[1], v_rot[2]
        else:
            vx, vy, vz, vr, vp, vy_ang = (None, None, None, None, None, None)

        row = [
            round(float(timestamp), 4),
            px, py, pz,
            tx, ty, tz,
            roll, pitch, yaw,
            round(float(cte), 4),
            round(float(he), 4),
            round(float(speed), 2),
            round(float(steer), 4),
            int(has_lock),
            round(float(ox), 4), 
            round(float(oy), 4), 
            round(float(oyaw), 4),
            # Vision Columns
            vx, vy, vz,
            vr, vp, vy_ang,
            round(float(theta_r), 4) if theta_r is not None else None
        ]
        self.writer.writerow(row)

    def close(self):
        if self.file:
            self.file.close()

class StanleyController:
    @staticmethod
    def calculate_steering(cross_track_error, heading_error, speed, k=0.8, epsilon=1e-5):
        cross_track_term = math.atan2(k * cross_track_error, abs(speed) + epsilon)
        return heading_error + cross_track_term
    
    @staticmethod
    def calculate_velocity(cross_track_error, heading_error, max_speed=1.0):
        error_magnitude = abs(cross_track_error) + abs(heading_error)
        speed_factor = 1.0 / (1.0 + error_magnitude)
        desired_speed = max_speed * speed_factor
        return max(0.2, desired_speed)

class GlobalPoseEstimator:
    def __init__(self, tag_map_path, tag_size_meters, ref_path_file="ref_path.csv"):
        self.tag_size = tag_size_meters
        self.tag_map = self._load_map(tag_map_path)
        self.world_corners = {} 
        self._generate_world_corners()
        
        # --- Vehicle Geometry (Car Body Frame) ---
        # Camera Offset relative to Car Base (Rear Axle Center)
        self.frontSlotY = 0.0
        self.frontSlotZ = 0.45
        self.frontSlotX = 3.85
        self.cameraX = 0.0954
        self.cameraY = 0.0
        self.cameraZ = 0.169114 + 0.375
        
        # Total distance from Rear Axle to Camera (approximate X-offset)
        self.cam_offset_x = self.frontSlotX + self.cameraX

        self.t_base_to_cam = np.array([
            self.cam_offset_x, 
            self.cameraY + self.frontSlotY, 
            self.cameraZ + self.frontSlotZ
        ])

        # Rotation from Car Body (Flu) to Camera Optical (Rub)
        self.R_car_to_cam = np.array([
            [0, -1,  0],
            [0,  0, -1],
            [1,  0,  0]
        ])

        # --- Localization State ---
        # Internal state [x, y, yaw_radians] (Global Frame)
        self.state = None 
        
        # Load Reference Path for fallback estimation
        self.ref_path = self._load_ref_path(ref_path_file)
        self.last_closest_idx = 0

    def _load_map(self, path):
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load tag map: {e}")
            return {}
            
    def _load_ref_path(self, path):
        try:
            data = []
            with open(path, 'r') as f:
                reader = csv.reader(f)
                header_skipped = False
                for idx, row in enumerate(reader):
                    # Only process every 10th row
                    if idx % 25 != 0:
                        continue
                    
                    # Simple heuristic to skip header if it contains text
                    try:
                        # Try to convert the first item to float to check if it's a data row
                        float(row[0])
                    except ValueError:
                        continue 
                    
                    try:
                        # FIX: Read index 1 (x) and 2 (y), not 0 and 1
                        if len(row) >= 3:
                            data.append([float(row[1]), float(row[2])])
                    except ValueError:
                        continue 
            
            if not data:
                print(f"Warning: Reference path {path} was empty or invalid.")
                return np.empty((0, 2))
                
            print(f"Loaded reference path with {len(data)} points.")
            return np.array(data)
        except Exception as e:
            print(f"Warning: Could not load ref path {path}: {e}")
            return np.empty((0, 2))

    def _generate_world_corners(self):
        s = self.tag_size / 2.0
        local_corners = np.array([
            [-s, -s, 0], [ s, -s, 0], [ s,  s, 0], [-s,  s, 0]
        ])
        
        for tag_id, pose_data in self.tag_map.items():
            if len(pose_data) != 7: continue
            tx, ty, tz, rx, ry, rz, angle = pose_data
            rot_vec = np.array([rx, ry, rz]) * angle
            R, _ = cv2.Rodrigues(rot_vec)
            t = np.array([tx, ty, tz])
            w_corners = np.dot(local_corners, R.T) + t
            self.world_corners[tag_id] = w_corners.astype(np.float32)

    def _update_odometry(self, speed_mps, steering_angle, dt):
        """
        Step 1: Odometry Update.
        Updates the state prediction based on vehicle kinematics.
        """
        if self.state is None: return

        x, y, yaw = self.state
        dist = speed_mps * dt
        
        # Kinematic Bicycle Model
        yaw += (dist * math.tan(steering_angle) / WHEEL_BASE)
        x += dist * math.cos(yaw)
        y += dist * math.sin(yaw)
        
        self.state = np.array([x, y, yaw])

    def _estimate_pose_from_lane(self, cte, he):
        """
        Step 3: Estimate Global Pose from Lane Data.
        Modified to search for the closest path point relative to the camera 
        (front of the vehicle) rather than the rear axle.
        """
        if self.state is None or len(self.ref_path) < 2:
            return False, None

        # 1. Get latest estimate (from Odometry, rear axle)
        est_x, est_y, est_yaw = self.state

        # --- NEW: Calculate Estimated Camera Position ---
        # Project forward by cam_offset_x along the current yaw
        cam_est_x = est_x + self.cam_offset_x * math.cos(est_yaw)
        cam_est_y = est_y + self.cam_offset_x * math.sin(est_yaw)

        # 2. Find Closest Point to the CAMERA (GLOBAL SEARCH ALWAYS)
        # Calculate distance to ALL points in the reference path from the camera
        all_distances = np.linalg.norm(self.ref_path - np.array([cam_est_x, cam_est_y]), axis=1)
        best_idx = np.argmin(all_distances)

        self.last_closest_idx = best_idx 
        
        # 3. Determine Path Segment (A -> B)
        # We need a line segment to project onto. Use (best-1 -> best) or (best -> best+1)
        if best_idx < len(self.ref_path) - 1:
            # Prefer forward segment
            pA = self.ref_path[best_idx]
            pB = self.ref_path[best_idx + 1]
        elif best_idx > 0:
            # Fallback to backward segment if at end
            pA = self.ref_path[best_idx - 1]
            pB = self.ref_path[best_idx]
        else:
            return False, None

        # 4. Project CAMERA Position onto Line Segment (Longitudinal Correction)
        # Vector A->B
        AB = pB - pA
        AB_len_sq = np.dot(AB, AB)
        if AB_len_sq < 1e-6: return False, None
        
        # Vector A -> Estimated Camera
        A_EstCam = np.array([cam_est_x, cam_est_y]) - pA
        
        # Scalar projection factor 't'
        t = np.dot(A_EstCam, AB) / AB_len_sq
        t = np.clip(t, 0.0, 1.0) # Clamp to segment
        
        # The "Projected Point" on the path (Longitudinally aligned with the Camera)
        p_on_line = pA + t * AB

        # 5. Calculate Heading of the Segment
        theta_r = math.atan2(AB[1], AB[0])

        # 6. Apply Lateral Correction (CTE)
        # New Global Heading
        global_yaw = theta_r - he
        
        # Calculate Sensor Position based on Projected Point + CTE
        # CTE direction is perpendicular to path heading
        x_front = pA[0] + cte * math.sin(theta_r)
        y_front = pA[1] - cte * math.cos(theta_r)

        # 7. Transform back to Rear Axle (The "Target" State)
        x_rear = x_front - self.cam_offset_x * math.cos(global_yaw)
        y_rear = y_front - self.cam_offset_x * math.sin(global_yaw)

        # 8. Update State by calculating Error and incrementing
        # Measure error: Target (Lane Calc) - Current (Odometry)
        error_x = x_rear - est_x
        error_y = y_rear - est_y
        error_yaw = global_yaw - est_yaw

        # Apply the error
        self.state[0] += error_x
        self.state[1] += error_y
        self.state[2] += error_yaw
        
        return True, theta_r
    
    def _estimate_pose_from_lane2(self, cte, he, lookahead_dist=0.0):
        if self.state is None or len(self.ref_path) < 2:
            return False, None

        est_x, est_y, est_yaw = self.state

        # Calculate Estimated PROJECTED Position
        total_offset = self.cam_offset_x + lookahead_dist
        proj_est_x = est_x + total_offset * math.cos(est_yaw)
        proj_est_y = est_y + total_offset * math.sin(est_yaw)
        p_est = np.array([proj_est_x, proj_est_y])

        # FIX 1: Local Search Window (Prevents map-snapping/jumps)
        search_window = 30 # Points to check forward/backward
        if hasattr(self, 'last_closest_idx') and self.last_closest_idx is not None:
            start_idx = max(0, self.last_closest_idx - search_window)
            end_idx = min(len(self.ref_path), self.last_closest_idx + search_window)
            window_indices = np.arange(start_idx, end_idx)
            
            distances = np.linalg.norm(self.ref_path[start_idx:end_idx] - p_est, axis=1)
            local_best = np.argmin(distances)
            best_idx = window_indices[local_best]
        else:
            # Global fallback
            all_distances = np.linalg.norm(self.ref_path - p_est, axis=1)
            best_idx = np.argmin(all_distances)

        self.last_closest_idx = best_idx 
        
        # FIX 2: Correct Segment Selection
        if best_idx == 0:
            pA, pB = self.ref_path[0], self.ref_path[1]
        elif best_idx == len(self.ref_path) - 1:
            pA, pB = self.ref_path[-2], self.ref_path[-1]
        else:
            # Check if we are between (best-1, best) or (best, best+1)
            dist_prev = np.linalg.norm(self.ref_path[best_idx - 1] - p_est)
            dist_next = np.linalg.norm(self.ref_path[best_idx + 1] - p_est)
            
            if dist_prev < dist_next:
                pA, pB = self.ref_path[best_idx - 1], self.ref_path[best_idx]
            else:
                pA, pB = self.ref_path[best_idx], self.ref_path[best_idx + 1]

        AB = pB - pA
        AB_len_sq = np.dot(AB, AB)
        if AB_len_sq < 1e-6: return False, None
        
        A_EstProj = p_est - pA
        t = np.dot(A_EstProj, AB) / AB_len_sq
        t = np.clip(t, 0.0, 1.0) 
        p_on_line = pA + t * AB

        theta_r = math.atan2(AB[1], AB[0])
        
        # NOTE: Verify if this should be theta_r - he (it is -he in your other method!)
        global_yaw = theta_r + he 
        
        x_front = p_on_line[0] + cte * math.sin(theta_r)
        y_front = p_on_line[1] - cte * math.cos(theta_r)

        x_rear = x_front - total_offset * math.cos(global_yaw)
        y_rear = y_front - total_offset * math.sin(global_yaw)

        error_x = x_rear - est_x
        error_y = y_rear - est_y
        error_yaw = global_yaw - est_yaw

        self.state[0] += error_x
        self.state[1] += error_y
        self.state[2] += error_yaw
        
        return True, theta_r

    def update(self, tags, camera_matrix, dist_coeffs, speed_mps, steering, dt, cte=None, he=None, has_lane_lock=False, lookahead_dist=0.0):
        """
        Main update loop.
        """
        
        # --- Step 1: Predict (Odometry)  ---
        # We always run this first so self.state reflects the motion since last frame
        self._update_odometry(speed_mps, steering, dt)

        # --- Step 2: Absolute Fix (AprilTags) ---
        tag_success = False
        vis_trans, vis_rot = None, None
        theta_r = None # Default value
        
        obj_points = []
        img_points = []
        found_tags = 0
        
        for tag in tags:
            tid = tag.tag_id
            if tid in self.world_corners:
                obj_points.extend(self.world_corners[tid])
                img_points.extend(tag.corners)
                found_tags += 1
        
        if found_tags >= 2:
            if dist_coeffs is None: dist_coeffs = np.zeros((4,1))
            success, rvec, tvec = cv2.solvePnP(np.array(obj_points), np.array(img_points), camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
            
            if success:
                R_world_to_cam, _ = cv2.Rodrigues(rvec)
                R_cam_to_world = R_world_to_cam.T
                P_cam_world = -R_cam_to_world @ tvec
                
                R_car_to_world = R_cam_to_world @ self.R_car_to_cam
                offset_in_world = R_car_to_world @ self.t_base_to_cam.reshape(3, 1)
                P_car_world = P_cam_world - offset_in_world
                
                sy = math.sqrt(R_car_to_world[0,0]**2 + R_car_to_world[1,0]**2)
                yaw = math.atan2(R_car_to_world[1,0], R_car_to_world[0,0]) if sy > 1e-6 else 0
                
                self.state = np.array([P_car_world[0,0], P_car_world[1,0], yaw])
                tag_success = True
                
                vis_trans = P_car_world.flatten()
                vis_rot = np.degrees([0, 0, yaw]) 
                
                # Update closest index on ref path so the next frame's lane search is close
                if len(self.ref_path) > 0:
                     dists = np.linalg.norm(self.ref_path - self.state[:2], axis=1)
                     self.last_closest_idx = np.argmin(dists)

        # --- Step 3: Lane Correction (if Tags failed) ---
        if not tag_success and has_lane_lock and cte is not None and he is not None:
             lane_success, theta_r = self._estimate_pose_from_lane2(cte, he, lookahead_dist)
             if lane_success:
                 vis_trans = np.array([self.state[0], self.state[1], 0.0]) 
                 vis_rot = np.degrees([0, 0, self.state[2]])

        # Fallback for visualization if we only have Odometry
        if not tag_success and not has_lane_lock and self.state is not None:
             vis_trans = np.array([self.state[0], self.state[1], 0.0])
             vis_rot = np.degrees([0, 0, self.state[2]])

        return tag_success, vis_trans, vis_rot, theta_r

    def estimate_pose2(self, tags, camera_matrix, dist_coeffs=None):
        return False, None, None

class WebotsLaneFollower:
    def __init__(self):
        # 1. Initialize Webots Driver
        self.driver = Driver()
        self.timestep = int(self.driver.getBasicTimeStep())
        if self.timestep == 0: self.timestep = 32
        self.logger = DataLogger("result.csv")

        # 2. Initialize Camera
        self.camera = self.driver.getDevice("camera")
        if self.camera:
            self.camera.enable(self.timestep)
            self.cam_width = self.camera.getWidth()
            self.cam_height = self.camera.getHeight()
            fov = self.camera.getFov()
            self.bev = BEVCalibrator(self.cam_width, self.cam_height, fov)
            self.at_detector = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
            self.tag_size_meters = 0.30
            self.camera_matrix = self.bev.K
            self.camera_params = (self.bev.K[0,0], self.bev.K[1,1], self.bev.K[0,2], self.bev.K[1,2])
        else:
            print("Error: Camera not found")

        self.gps = self.driver.getDevice("gps")
        if self.gps:
            self.gps.enable(self.timestep)
            print("GPS Enabled.")
        else:
            print("Warning: GPS device not found (check name 'gps').")

        self.imu = self.driver.getDevice("inertial unit")
        if self.imu:
            self.imu.enable(self.timestep)
            print("IMU Enabled.")
        else:
            print("Warning: IMU device not found (check name 'inertial unit').")

        # 3. Initialize Keyboard
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)

        # 4. Control State
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.lane_thresholds = (190, 255) 
        self.calibrated = False
        
        self.bev_calibrated = False
        self.M = None
        self.bev_center = None
        self.bev_ppm = None
        
        self.pose_estimator = GlobalPoseEstimator(TAG_MAP_FILE, self.tag_size_meters)

    def getspeedsteer(self):
        return self.driver.getCurrentSpeed(), self.driver.getSteeringAngle()

    def calibrate_lane_thresholds(self, frame_bgr, exclusion_mask=None, matrix=None):
        print("Calibrating lane thresholds...")
        if matrix is not None:
             h, w = frame_bgr.shape[:2]
             warped = cv2.warpPerspective(frame_bgr, matrix, (w, h))
        else:
            warped = frame_bgr

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        if exclusion_mask is not None:
            if matrix is not None and exclusion_mask.shape == frame_bgr.shape[:2]:
                 exclusion_mask = cv2.warpPerspective(exclusion_mask, matrix, (w, h))
            gray = cv2.bitwise_and(gray, exclusion_mask)

        H_img = warped.shape[0]
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist[0] = 0 
        hist_normalized = cv2.normalize(hist, None, 0, H_img-1, cv2.NORM_MINMAX)

        peaks = np.where((hist_normalized[1:-1] > hist_normalized[:-2]) & 
                         (hist_normalized[1:-1] > hist_normalized[2:]))[0] + 1
        
        filtered_peaks = []
        for p in peaks:
            if all(abs(p - fp) > 20 for fp in filtered_peaks):
                filtered_peaks.append(p)

        peaks = sorted(filtered_peaks, key=lambda x: hist_normalized[x], reverse=True)
        used_peak = max(peaks) if peaks else None

        if used_peak is not None:
            span = 30
            lower = max(0, used_peak - span)
            upper = min(255, used_peak + span)
            self.lane_thresholds = (int(lower), int(upper))
            self.calibrated = True
            print(f"Calibration Successful. Peak: {used_peak}, Range: {self.lane_thresholds}")

    def calculate_cte(ppm, cx, cy, m, b):
        v_axle = (cx, cy)
        if m < 1e-6:
            x_int, y_int = cy, b
        else:
            perp_m = -1 / m
            perp_b = cy - perp_m * cx
            x_int = (perp_b - b) / (m - perp_m)
            y_int = m * x_int + b
        
        int_point = (x_int, y_int)
        error_pixels = math.hypot(x_int - cx, y_int - cy)

        x_at_axle_line = (cy - b)/m if abs(m) > 1e-6 else float('inf')

        if x_at_axle_line < cx:
            error_pixels = -error_pixels

        return error_pixels, int_point
        
    def process_vision_pipeline(self, warped_img, cam_center, ppm, exclusion_mask=None, lookahead_m=0.0):
        debug_frame = warped_img.copy()
        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        if exclusion_mask is not None:
            gray = cv2.bitwise_and(gray, exclusion_mask)
        
        h, w = gray.shape
        roi_mask = np.zeros_like(gray)
        roi_points = np.array([
            [360, 0],                         
            [910, 0],                         
            [int(w * 0.51), int(450)], 
            [int(w * 0.49), int(450)], 
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, [roi_points], 255)
        gray = cv2.bitwise_and(gray, roi_mask)
        cv2.polylines(debug_frame, [roi_points], isClosed=True, color=(255, 0, 0), thickness=2)

        lower_thresh, upper_thresh = self.lane_thresholds
        mask_binary = cv2.inRange(gray, lower_thresh, upper_thresh)
        
        kernel = np.ones((3,3), np.uint8)
        morphed = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug_frame, contours, -1, (0, 255, 0), 1)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < 550]

        path_points = []
        for cnt in valid_contours:
            center = _contour_center(cnt)
            if center:
                path_points.append(center)
        
        path_points.sort(key=lambda p: p[1], reverse=True)

        cte = 0.0
        proj_cte = 0.0 # NEW: Projected CTE
        he = 0.0
        has_lock = False

        if len(path_points) >= 2:
            p_close = path_points[0]
            p_far = path_points[1] 
            cv2.line(debug_frame, p_close, p_far, (0, 255, 255), 2)
            
            vx, vy = cam_center
            
            # Project camera center forward in BEV image space (Y goes up/negative)
            vy_proj = vy - (lookahead_m * ppm) 

            dx = p_far[0] - p_close[0]
            dy = p_far[1] - p_close[1]
            path_angle = math.atan2(-dy, dx) 
            he = (math.pi / 2) - path_angle
            path_len = math.hypot(dx, dy)
            
            # --- 1. Base CTE (for Stanley Control) ---
            cross_prod = dx * (vy - p_close[1]) - dy * (vx - p_close[0])
            cte_pixels = cross_prod / path_len
            cte = -cte_pixels / ppm
            
            # --- 2. Projected CTE (for Pose Estimation) ---
            proj_cross_prod = dx * (vy_proj - p_close[1]) - dy * (vx - p_close[0])
            proj_cte_pixels = proj_cross_prod / path_len
            proj_cte = -proj_cte_pixels / ppm
            
            has_lock = True
        
            # Visualization for Base CTE
            t = ((vx - p_close[0]) * dx + (vy - p_close[1]) * dy) / (path_len * path_len)
            proj_x = int(p_close[0] + t * dx)
            proj_y = int(p_close[1] + t * dy)
            cv2.line(debug_frame, (int(vx), int(vy)), (proj_x, proj_y), (0, 0, 255), 2)
            cv2.circle(debug_frame, (proj_x, proj_y), 4, (0, 0, 255), -1)

            # Visualization for Projected CTE
            t_proj = ((vx - p_close[0]) * dx + (vy_proj - p_close[1]) * dy) / (path_len * path_len)
            proj_x2 = int(p_close[0] + t_proj * dx)
            proj_y2 = int(p_close[1] + t_proj * dy)
            cv2.line(debug_frame, (int(vx), int(vy_proj)), (proj_x2, proj_y2), (255, 0, 255), 2)
            cv2.circle(debug_frame, (int(vx), int(vy_proj)), 4, (255, 0, 255), -1)

            cv2.putText(debug_frame, f"CTE: {cte:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(debug_frame, f"Proj CTE: {proj_cte:.2f}m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        return cte, proj_cte, he, has_lock, debug_frame

    def run(self):
        print("Starting Webots Lane Follower...")

        odom_x = 0.0
        odom_y = 0.0
        odom_yaw = 0.0 
        
        if self.gps:
            self.driver.step() 
            initial_gps = self.gps.getValues()
            odom_x = initial_gps[0]
            odom_y = initial_gps[1]
            if self.imu:
                 odom_yaw = self.imu.getRollPitchYaw()[2]
        
        while self.driver.step() != -1:
            key = self.keyboard.getKey()
            manual_override = False
            
            # Reset vision pose variables for this frame
            vis_trans = None
            vis_rot = None
            
            if key == Keyboard.UP:
                self.current_speed += 1.0
                manual_override = True
            elif key == Keyboard.DOWN:
                self.current_speed -= 1.0
                manual_override = True
            elif key == ord(' '):
                self.current_speed = 0.0
                self.current_steering = 0.0
                manual_override = True

            # --- ODOMETRY LOGIC ---
            raw_speed_kmh, raw_steer = self.getspeedsteer()
            noisy_speed_kmh = raw_speed_kmh + np.random.normal(0, SPEED_NOISE_STD)
            noisy_steer = raw_steer + np.random.normal(0, STEER_NOISE_STD)
            speed_mps = noisy_speed_kmh / 3.6
            
            if not math.isnan(raw_speed_kmh) and not math.isnan(raw_steer):
                step_duration_sec = self.timestep / 1000.0
                dist = speed_mps * step_duration_sec
                if abs(dist) > 0.00001:
                    odom_yaw += (dist * math.tan(noisy_steer) / WHEEL_BASE)
                    odom_x += dist * math.cos(odom_yaw)
                    odom_y -= dist * math.sin(odom_yaw)

            # --- Image Capture ---
            raw_image = self.camera.getImage()
            if raw_image:
                img_np = np.frombuffer(raw_image, np.uint8).reshape((self.cam_height, self.cam_width, 4))
                frame_bgr = img_np[:, :, :3].copy()
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                # --- Tag Detection & Pose ---
                tags = self.at_detector.detect(gray, estimate_tag_pose=False, camera_params=self.camera_params, tag_size=self.tag_size_meters)
                tag_centers = {tag.tag_id: tag.center for tag in tags}
                
                h, w = frame_bgr.shape[:2]
                tag_mask = np.ones((h, w), dtype=np.uint8) * 255
                MASK_EXPANSION = 1.4
                if tags:
                    for tag in tags:
                        center = tag.center
                        corners = tag.corners
                        expanded_corners = center + (corners - center) * MASK_EXPANSION
                        pts = expanded_corners.astype(np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(tag_mask, [pts], 0)

                # --- BEV Calibration Logic ---
                required_ids = {0, 1, 2, 3}
                present_ids = set(tag_centers.keys())
                should_run_calibration = required_ids.issubset(present_ids)

                warped = None
                matrix = None
                cam_center = None
                ppm = None
                
                if should_run_calibration:
                    warped, matrix, cam_center, ppm, is_valid = self.bev.process(frame_bgr, tag_centers)
                    if is_valid:
                        self.M = matrix
                        self.bev_center = cam_center
                        self.bev_ppm = ppm
                        self.bev_calibrated = True
                elif self.bev_calibrated:
                    matrix = self.M
                    cam_center = self.bev_center
                    ppm = self.bev_ppm
                    warped = cv2.warpPerspective(frame_bgr, matrix, (w, h))

                if key == ord('C'):
                     self.calibrate_lane_thresholds(frame_bgr, tag_mask, matrix)

                # --- Main Lane Logic & Data Logging ---
                if self.bev_calibrated and warped is not None:
                    warped_mask = cv2.warpPerspective(tag_mask, matrix, (w, h))
                    
                    # Pass the LOOKAHEAD param and unpack the newly returned proj_cte
                    cte, proj_cte, he, has_lock, debug_img = self.process_vision_pipeline(
                        warped, cam_center, ppm, warped_mask, lookahead_m=POSE_LOOKAHEAD_M
                    )
                    
                    # --- GLOBAL POSE ESTIMATOR UPDATE ---
                    # Feed the PROJ_CTE to the pose estimator, along with the lookahead distance
                    vis_success, vis_trans, vis_rot, theta_r = self.pose_estimator.update(
                        tags, self.camera_matrix, None, 
                        speed_mps, noisy_steer, self.timestep/1000.0,
                        cte=proj_cte, he=he, has_lane_lock=has_lock, lookahead_dist=POSE_LOOKAHEAD_M
                    )

                    # --- STANLEY CONTROLLER ---
                    # Feed the STANDARD CTE to the controller 
                    if has_lock and not manual_override:
                        speed = StanleyController.calculate_velocity(cte, he, CRUISING_SPEED)
                        steer = StanleyController.calculate_steering(cte, he, speed, k=STANLEY_K)
                        self.current_steering = steer
                        self.current_speed = speed

                    # ==========================================
                    #  VISUALIZE STEERING ON DEBUG FRAME
                    # ==========================================
                    if self.bev_center is not None:
                        cx, cy = int(self.bev_center[0]), int(self.bev_center[1])
                        line_length = self.current_speed  
                        end_x = int(cx - line_length * math.sin(self.current_steering))
                        end_y = int(cy - line_length * math.cos(self.current_steering))
                        
                        cv2.line(debug_img, (cx, cy), (end_x, end_y), (0, 0, 255), 3)
                        cv2.circle(debug_img, (cx, cy), 5, (255, 0, 0), -1)
                        cv2.circle(debug_img, (end_x, end_y), 5, (0, 0, 255), -1)
                        cv2.putText(debug_img, f"Steer: {self.current_steering:.3f}", (10, 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # ==========================================

                    track_pos_est = (None, None, None)
                    gps_pos = None
                    car_rot = None
                    
                    if self.gps and self.imu:
                        gps_pos = self.gps.getValues() 
                        rpy = self.imu.getRollPitchYaw()
                        car_rot = np.degrees(rpy)

                    # --- Log Data ---
                    self.logger.log(
                        timestamp=self.driver.getTime(),
                        pos=gps_pos,
                        track_pos=track_pos_est,
                        rot=car_rot,
                        cte=cte,
                        he=he,
                        speed=self.current_speed,
                        steer=self.current_steering,
                        has_lock=has_lock,
                        odom=(odom_x, odom_y, odom_yaw),
                        vision_pose=(vis_trans, vis_rot),
                        theta_r=theta_r # <--- Pass the new heading tracking variable
                    )

                    cv2.imshow("BEV Driver", debug_img)
                    cv2.imshow("Raw Camera", frame_bgr)
                else:
                    cv2.putText(frame_bgr, "Waiting for Tags 0,1,2,3...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.imshow("BEV Driver", frame_bgr)
                    
                cv2.waitKey(1)

            self.driver.setCruisingSpeed(self.current_speed)
            self.driver.setSteeringAngle(self.current_steering)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = WebotsLaneFollower()
    controller.run()