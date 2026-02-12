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
STANLEY_K = 1.5
TAG_MAP_FILE = "output2.yaml" 

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
        
        # Updated Header with Track Position Estimates AND Odometry
        header = [
            "timestamp", 
            "pos_x", "pos_y", "pos_z",       # Car GPS Position
            "track_x", "track_y", "track_z", # Calculated Track Center Position
            "roll", "pitch", "yaw",          # Car Orientation (IMU)
            "cross_track_error", "heading_error", 
            "speed_cmd", "steering_cmd", "has_lock",
            "odom_x", "odom_y", "odom_yaw"   # <--- ADDED ODOMETRY COLUMNS
        ]
        self.writer.writerow(header)
        print(f"Data logger initialized. Writing to: {self.filename}")

    def log(self, timestamp, pos, track_pos, rot, cte, he, speed, steer, has_lock, odom):
        px, py, pz = pos if pos is not None else (None, None, None)
        tx, ty, tz = track_pos if track_pos is not None else (None, None)
        roll, pitch, yaw = rot if rot is not None else (None, None, None)
        
        # Unpack Odometry
        ox, oy, oyaw = odom

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
            round(float(ox), 4), # Odom X
            round(float(oy), 4), # Odom Y
            round(float(oyaw), 4) # Odom Yaw
        ]
        self.writer.writerow(row)

    def close(self):
        if self.file:
            self.file.close()

class StanleyController:
    @staticmethod
    def calculate_steering(cross_track_error, heading_error, speed, k=0.8, epsilon=1e-5):
        denom = abs(speed*0.277778) + epsilon
        cross_track_term = math.atan2(k * -cross_track_error, denom)
        return heading_error + cross_track_term
    
    @staticmethod
    def calculate_velocity(cross_track_error, heading_error, max_speed=1.0):
        error_magnitude = abs(cross_track_error) + abs(heading_error)
        speed_factor = 1.0 / (1.0 + error_magnitude)
        desired_speed = max_speed * speed_factor
        return max(0.2, desired_speed)

class GlobalPoseEstimator:
    def __init__(self, tag_map_path, tag_size_meters):
        self.tag_size = tag_size_meters
        self.tag_map = self._load_map(tag_map_path)
        self.world_corners = {} 
        self._generate_world_corners()

    def _load_map(self, path):
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load tag map: {e}")
            return {}

    def _generate_world_corners(self):
        s = self.tag_size / 2.0
        local_corners = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
        for tag_id, pose_data in self.tag_map.items():
            if len(pose_data) != 7: continue
            tx, ty, tz, rx, ry, rz, angle = pose_data
            rot_vec = np.array([rx, ry, rz]) * angle
            R, _ = cv2.Rodrigues(rot_vec)
            t = np.array([tx, ty, tz])
            w_corners = np.dot(local_corners, R.T) + t
            self.world_corners[tag_id] = w_corners.astype(np.float32)

    def estimate_pose(self, tags, camera_matrix, dist_coeffs=None):
        obj_points = []
        img_points = []
        found_tags = 0
        for tag in tags:
            tid = tag.tag_id
            if tid in self.world_corners:
                obj_points.extend(self.world_corners[tid])
                img_points.extend(tag.corners)
                found_tags += 1
        
        if found_tags < 2: return False, None, None
        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)
        if dist_coeffs is None: dist_coeffs = np.zeros((4,1))

        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
        if not success: return False, None, None

        R, _ = cv2.Rodrigues(rvec)
        R_inv = R.T
        cam_pos_world = -R_inv @ tvec
        
        # Calculate Euler Angles
        sy = math.sqrt(R_inv[0,0] * R_inv[0,0] +  R_inv[1,0] * R_inv[1,0])
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(R_inv[2,1] , R_inv[2,2])
            pitch = math.atan2(-R_inv[2,0], sy)
            yaw = math.atan2(R_inv[1,0], R_inv[0,0])
        else:
            roll = math.atan2(-R_inv[1,2], R_inv[1,1])
            pitch = math.atan2(-R_inv[2,0], sy)
            yaw = 0
        return True, cam_pos_world.flatten(), np.degrees([roll, pitch, yaw])

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
        """Helper to get current real speed and steering angle from Webots"""
        # getCurrentSpeed() returns speed in km/h or m/s depending on model, 
        # but usually m/s in new nodes, km/h in Driver.
        # Driver.getCurrentSpeed() returns km/h usually. 
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

    def process_vision_pipeline(self, warped_img, cam_center, ppm, exclusion_mask=None):
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
        he = 0.0
        has_lock = False

        if len(path_points) >= 2:
            p_close = path_points[0]
            p_far = path_points[1] 
            cv2.line(debug_frame, p_close, p_far, (0, 255, 255), 2)
            
            dx = p_far[0] - p_close[0]
            dy = p_far[1] - p_close[1]
            path_angle = math.atan2(-dy, dx) 
            he = (math.pi / 2) - path_angle

            vx, vy = cam_center
            if abs(dx) < 1e-5: 
                cte_pixels = vx - p_close[0]
                proj_pt = (p_close[0], int(vy))
            else:
                m = dy / dx
                c_val = p_close[1] - (m * p_close[0])
                cte_pixels = (m * vx - vy + c_val) / math.sqrt(m**2 + 1)
                epsilon = 1e-5
                m_perp = -1/(m+epsilon)
                c_perp = vy - (m_perp * vx)
                inter_x = (c_perp - c_val) / (m - m_perp)
                inter_y = m * inter_x + c_val
                proj_pt = (int(inter_x), int(inter_y))
            
            cte = cte_pixels / ppm
            has_lock = True
            
            cv2.line(debug_frame, (int(vx), int(vy)), proj_pt, (0, 0, 255), 2)
            cv2.circle(debug_frame, proj_pt, 4, (0, 0, 255), -1)
            cv2.putText(debug_frame, f"CTE: {cte:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(debug_frame, f"HE: {math.degrees(he):.1f}deg", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return cte, he, has_lock, debug_frame

    def run(self):
        print("Starting Webots Lane Follower...")

        # --- ODOMETRY INITIALIZATION ---
        odom_x = 0.0
        odom_y = 0.0
        odom_yaw = 0.0 # Radians
        last_left_pos = 0.0
        
        # If possible, initialize Odom X/Y with initial GPS to match coordinates
        if self.gps:
            self.driver.step() # Step once to get sensor data
            initial_gps = self.gps.getValues()
            odom_x = initial_gps[0]
            odom_y = initial_gps[1]
            # odom_z = initial_gps[2] # We usually ignore Z for 2D odometry
            # Note: We assume initial yaw is 0 or needs to be fetched from IMU
            if self.imu:
                 odom_yaw = self.imu.getRollPitchYaw()[2]
        
        while self.driver.step() != -1:
            key = self.keyboard.getKey()
            manual_override = False
            
            # --- Input Handling ---
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
            # 1. Get raw values
            raw_speed_kmh, raw_steer = self.getspeedsteer()
            
            # 2. Add Noise
            # np.random.normal(mean, standard_deviation)
            noisy_speed_kmh = raw_speed_kmh + np.random.normal(0, SPEED_NOISE_STD)
            noisy_steer = raw_steer + np.random.normal(0, STEER_NOISE_STD)
            speed_mps = noisy_speed_kmh / 3.6
            
            # 3. Ensure values are valid numbers
            if not math.isnan(raw_speed_kmh) and not math.isnan(raw_steer):
                # 4. Get current wheel rotation (in radians) - approximating linear distance here
                # Distance = speed * time
                step_duration_sec = self.timestep / 1000.0
                
                # Calculate Distance Traveled this step
                dist = speed_mps * step_duration_sec
                
                # 5. Only update if the car has actually moved
                if abs(dist) > 0.00001:
                    # Update orientation (Yaw) using Bicycle Model
                    # yaw_new = yaw_old + (dist / L) * tan(delta)
                    odom_yaw += (dist * math.tan(noisy_steer) / WHEEL_BASE)
                    
                    # Update position (X and Y)
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
                
                # --- Create Exclusion Mask for Lane Detector ---
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

                    # Update global pose for visualization (optional)
                    self.pose_estimator.estimate_pose(tags, self.camera_matrix)

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
                    cte, he, has_lock, debug_img = self.process_vision_pipeline(warped, cam_center, ppm, warped_mask)
                    
                    if has_lock and not manual_override:
                        speed = StanleyController.calculate_velocity(cte, he, CRUISING_SPEED)
                        steer = StanleyController.calculate_steering(cte, he, speed, k=STANLEY_K)
                        if abs(self.current_steering - steer) > 0.1:
                            self.current_steering = self.current_steering + (0.01 * (steer/abs(steer)))
                        else:
                            self.current_steering = max(-0.5, min(0.5, steer))
                        self.current_speed = speed
                    
                    track_pos_est = (None, None, None)
                    gps_pos = None
                    car_rot = None
                    
                    if self.gps and self.imu and has_lock:
                        # 1. Get raw sensor data
                        gps_pos = self.gps.getValues() # [x, y, z]
                        rpy = self.imu.getRollPitchYaw() # [roll, pitch, yaw]
                        car_rot = np.degrees(rpy)
                        
                        car_yaw = rpy[2] # Global Yaw in radians
                        car_x, car_y, car_z = gps_pos
                        
                        # Calculate Perpendicular logic for CTE visualization
                        track_heading = car_yaw - he
                        # t_x = car_x - cte * math.cos(track_heading - math.pi/2)
                        # t_z = car_z - cte * math.sin(track_heading - math.pi/2)
                        # t_y = car_y # Webots Y is up/down usually, Z/X is ground plane. 
                        # NOTE: Standard Webots ENU: X, Y, Z. But often Y is vertical.
                        # Assuming X=Right, Z=Back/Front, Y=Up.
                        
                        # Simplified viz calc:
                        t_x = car_x - cte * math.cos(track_heading - math.pi/2)
                        t_z = car_z - cte * math.sin(track_heading - math.pi/2)
                        t_y = car_y - cte * math.sin(track_heading - math.pi/2)
                        track_pos_est = (t_x, t_y, t_z)

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
                        odom=(odom_x, odom_y, odom_yaw) # Pass odometry tuple
                    )

                    # Visualization
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