import cv2
import numpy as np
import math
import yaml
import csv
from pupil_apriltags import Detector

# Webots Imports
from vehicle import Driver
from controller import Keyboard

# Local Imports
from bev_calibrator import BEVCalibrator

# --- Configuration ---
MAX_SPEED_WEBOTS = 50.0 
CRUISING_SPEED = 100.0 # You may be able to safely increase this now with the BLF controller
STANLEY_K = 10.0
TAG_MAP_FILE = "output3.yaml" 
FRONT_SLOT_Y = 0.0
FRONT_SLOT_Z = 0.45
FRONT_SLOT_X = 3.85
CAMERA_X = 0.0954
CAMERA_Y = 0.0
CAMERA_Z = 0.169114
POSE_LOOKAHEAD_M = -3.5 # Meters to project ahead for pose estimation

# --- Performance & Execution Options ---
TAG_DETECT_INTERVAL = 3  # Detect AprilTags every N frames once calibrated
SHOW_GUI = True          
GUI_SKIP_INTERVAL = 2    
TAG_MASK_TTL_FRAMES = 8  # Keep masking tag area for N frames after detection drops

# --- Odometry Configuration ---
WHEEL_RADIUS = 0.40  # Meters (Approx for BMW X5 in Webots)
WHEEL_BASE = 2.995   # Meters (Distance between front and rear axles)
SPEED_NOISE_STD = 0.5 # Standard deviation for speed noise (km/h)
STEER_NOISE_STD = 0.005 # Standard deviation for steering noise (radians)

def _contour_center(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0: return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

class DataLogger:
    def __init__(self, filename="result.csv"):
        self.filename = filename
        self.file = open(self.filename, mode='w', newline='')
        self.writer = csv.writer(self.file)
        
        header = [
            "timestamp", 
            "gps_x", "gps_y", "gps_z",       
            "track_x", "track_y", "track_z", 
            "roll", "pitch", "yaw",          
            "cross_track_error", "heading_error", 
            "speed_cmd", "steering_cmd", "has_lock",
            "odom_x", "odom_y", "odom_yaw",  
            "vis_x", "vis_y", "vis_z",       
            "vis_roll", "vis_pitch", "vis_yaw", 
            "theta_r",                        
            "tag_detected"
        ]
        self.writer.writerow(header)

    def log(self, timestamp, pos, track_pos, rot, cte, he, speed, steer, has_lock, odom, vision_pose, theta_r, tag_detected):
        px, py, pz = pos if pos is not None else (None, None, None)
        
        if track_pos is not None and len(track_pos) == 3:
            tx, ty, tz = track_pos
        else:
            tx, ty, tz = None, None, None

        roll, pitch, yaw = rot if rot is not None else (None, None, None)
        ox, oy, oyaw = odom

        if vision_pose and vision_pose[0] is not None:
            v_trans, v_rot = vision_pose
            vx, vy, vz = v_trans[0], v_trans[1], v_trans[2]
            vr, vp, vy_ang = v_rot[0], v_rot[1], v_rot[2]
        else:
            vx, vy, vz, vr, vp, vy_ang = (None, None, None, None, None, None)

        row = [
            round(timestamp, 4),
            px, py, pz,
            tx, ty, tz,
            roll, pitch, yaw,
            round(cte, 4),
            round(he, 4),
            round(speed, 2),
            round(steer, 4),
            int(has_lock),
            round(ox, 4), 
            round(oy, 4), 
            round(oyaw, 4),
            vx, vy, vz,
            vr, vp, vy_ang,
            round(theta_r, 4) if theta_r is not None else None,
            int(tag_detected)
        ]
        self.writer.writerow(row)

    def close(self):
        if self.file:
            self.file.flush()
            self.file.close()

class BarrierLyapunovController:
    @staticmethod
    def calculate_steering(alpha, heading_error, speed_mps, rho=4.0, alpha_bar_deg=33.0, wheelbase=2.995, kb=1.0, epsilon=1e-6):
        """
        Calculates the steering angle using a Barrier Lyapunov Function to enforce FOV constraints.
        Matches the mathematical algebraic cancellation and kinematic mapping from simifac2026.m.
        """
        alpha_bar = math.radians(alpha_bar_deg)

        # Prevent strict constraint violations
        if abs(alpha) >= alpha_bar:
            alpha = math.copysign(alpha_bar - 1e-4, alpha)

        # Safe bounds to prevent division by zero
        v_safe = max(abs(speed_mps), 1e-3)
        rho_safe = max(rho, 1e-3)

        # Calculate barrier components
        denom = (math.sin(alpha_bar)**2) - (math.sin(alpha)**2)
        Gamma = kb * (math.sin(alpha) * math.cos(alpha)) / (denom + epsilon)
        
        # Algebraic cancellation of sin(alpha) (from MATLAB term1_div_Gamma)
        term1_div_Gamma = (heading_error + Gamma) * v_safe * (denom + epsilon) / (rho_safe * kb * math.cos(alpha))
        omega = term1_div_Gamma + Gamma

        # Map yaw rate back to steering angle using atan2 kinematic mapping
        steering_angle = math.atan2(wheelbase * omega, v_safe)
        
        return steering_angle


class StanleyController:
    @staticmethod
    def calculate_steering(cross_track_error, heading_error, speed, k=1.0, epsilon=1e-5):
        cross_track_term = math.atan2(k * cross_track_error, abs(speed) + epsilon)
        return heading_error + cross_track_term
    
    @staticmethod
    def calculate_velocity(cross_track_error, heading_error, max_speed=1.0):
        error_magnitude = abs(cross_track_error) + abs(heading_error)
        speed_factor = 1.0 / (1.0 + error_magnitude)
        desired_speed = max_speed * speed_factor
        return max(0.2, desired_speed)


class GlobalPoseEstimator:
    def __init__(self, tag_map_path, tag_size_meters, ref_path_file="ref_path4.csv"):
        self.tag_size = tag_size_meters
        self.tag_map = self._load_map(tag_map_path)
        self.world_corners = {} 
        self._generate_world_corners()
        
        self.frontSlotY = 0.0
        self.frontSlotZ = 0.45
        self.frontSlotX = 3.85
        self.cameraX = 0.0954
        self.cameraY = 0.0
        self.cameraZ = 0.169114 + 0.375
        self.theta_r = 0.0
        
        self.cam_offset_x = self.frontSlotX + self.cameraX

        self.t_base_to_cam = np.array([
            self.cam_offset_x, 
            self.cameraY + self.frontSlotY, 
            self.cameraZ + self.frontSlotZ
        ])

        self.R_car_to_cam = np.array([
            [0, -1,  0],
            [0,  0, -1],
            [1,  0,  0]
        ])

        self.state = None 
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
                for idx, row in enumerate(reader):
                    if idx % 15 != 0:
                        continue
                    try:
                        float(row[0])
                    except ValueError:
                        continue 
                    
                    try:
                        if len(row) >= 3:
                            data.append([float(row[1]), float(row[2])])
                    except ValueError:
                        continue 
            return np.array(data) if data else np.empty((0, 2))
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
        if self.state is None: return

        x, y, yaw = self.state
        dist = speed_mps * dt
        
        yaw += (dist * math.tan(-steering_angle) / WHEEL_BASE) 
        x += dist * math.cos(yaw)
        y += dist * math.sin(yaw)
        
        self.state[0] = x
        self.state[1] = y
        self.state[2] = yaw

    def _estimate_pose_from_lane2(self, cte, he, lookahead_dist=0.0):
        if self.state is None or len(self.ref_path) < 2:
            return False, None

        est_x, est_y, est_yaw = self.state
        p_est = np.array([est_x, est_y])

        search_window = 30 
        if hasattr(self, 'last_closest_idx') and self.last_closest_idx is not None:
            start_idx = max(0, self.last_closest_idx - search_window)
            end_idx = min(len(self.ref_path), self.last_closest_idx + search_window)
            
            sq_distances = np.sum((self.ref_path[start_idx:end_idx] - p_est)**2, axis=1)
            best_idx = start_idx + np.argmin(sq_distances)
        else:
            sq_distances = np.sum((self.ref_path - p_est)**2, axis=1)
            best_idx = np.argmin(sq_distances)

        self.last_closest_idx = best_idx 
        
        if best_idx == 0:
            pA, pB = self.ref_path[0], self.ref_path[1]
        elif best_idx == len(self.ref_path) - 1:
            pA, pB = self.ref_path[-2], self.ref_path[-1]
        else:
            dist_prev_sq = np.sum((self.ref_path[best_idx - 1] - p_est)**2)
            dist_next_sq = np.sum((self.ref_path[best_idx + 1] - p_est)**2)
            if dist_prev_sq < dist_next_sq:
                pA, pB = self.ref_path[best_idx - 1], self.ref_path[best_idx]
            else:
                pA, pB = self.ref_path[best_idx], self.ref_path[best_idx + 1]

        AB = pB - pA
        AB_len_sq = np.dot(AB, AB)
        if AB_len_sq < 1e-6: return False, None
        
        A_EstProj = p_est - pA
        t = np.dot(A_EstProj, AB) / AB_len_sq
        t = np.clip(t, 0.0, 1.0) 
        x_ref, y_ref = pA + t * AB  

        theta_r = math.atan2(AB[1], AB[0])
        self.theta_r = theta_r 
        
        global_yaw = theta_r + he
        
        x_v = x_ref + cte * math.sin(theta_r)
        y_v = y_ref - cte * math.cos(theta_r)

        self.state[0] = x_v
        self.state[1] = y_v
        self.state[2] = global_yaw
        
        return True, theta_r

    def update(self, tags, camera_matrix, dist_coeffs, speed_mps, steering, dt, cte=None, he=None, has_lane_lock=False, lookahead_dist=0.0):
        self._update_odometry(speed_mps, steering, dt)

        tag_success = False
        vis_trans, vis_rot = None, None
        theta_r = None
        
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
            success, rvec, tvec, _ = cv2.solvePnPRansac(np.array(obj_points), np.array(img_points), camera_matrix, dist_coeffs)
            
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
                
                if len(self.ref_path) > 0:
                     sq_dists = np.sum((self.ref_path - self.state[:2])**2, axis=1)
                     self.last_closest_idx = np.argmin(sq_dists)

        if not tag_success and has_lane_lock and cte is not None and he is not None:
             lane_success, theta_r = self._estimate_pose_from_lane2(cte, he, lookahead_dist)
             if lane_success:
                 vis_trans = np.array([self.state[0], self.state[1], 0.0]) 
                 vis_rot = np.degrees([0, 0, self.state[2]])

        if not tag_success and not has_lane_lock and self.state is not None:
             vis_trans = np.array([self.state[0], self.state[1], 0.0])
             vis_rot = np.degrees([0, 0, self.state[2]])

        return tag_success, vis_trans, vis_rot, theta_r


class WebotsLaneFollower:
    def __init__(self):
        self.driver = Driver()
        self.timestep = int(self.driver.getBasicTimeStep())
        if self.timestep == 0: self.timestep = 32
        self.logger = DataLogger("result.csv")
        self.video_writer = None # Setup explicitly for OpenCV recording

        self.camera = self.driver.getDevice("camera")
        if self.camera:
            self.camera.enable(self.timestep)
            self.cam_width = self.camera.getWidth()
            self.cam_height = self.camera.getHeight()
            fov = self.camera.getFov()
            self.bev = BEVCalibrator(self.cam_width, self.cam_height, fov)
            self.at_detector = Detector(families="tag36h11", nthreads=4, quad_decimate=2.0)
            self.tag_size_meters = 0.30
            self.camera_matrix = self.bev.K
            self.camera_params = (self.bev.K[0,0], self.bev.K[1,1], self.bev.K[0,2], self.bev.K[1,2])
        else:
            print("Error: Camera not found")

        self.gps = self.driver.getDevice("gps")
        if self.gps: self.gps.enable(self.timestep)

        self.imu = self.driver.getDevice("inertial unit")
        if self.imu: self.imu.enable(self.timestep)

        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)

        self.current_speed = 0.0
        self.current_steering = 0.0
        self.lane_thresholds = (190, 255) 
        self.calibrated = False
        
        self.bev_calibrated = False
        self.M = None
        self.bev_center = None
        self.bev_ppm = None
        
        self.morph_kernel = np.ones((3,3), np.uint8)
        self.pose_estimator = GlobalPoseEstimator(TAG_MAP_FILE, self.tag_size_meters)
        self.cached_tags = []
        self.active_tag_polys = []
        self.last_valid_cte = None

    def getspeedsteer(self):
        return self.driver.getCurrentSpeed(), self.driver.getSteeringAngle()

    def calibrate_lane_thresholds(self, frame_bgr, exclusion_mask=None, matrix=None):
        if matrix is not None:
             h, w = frame_bgr.shape[:2]
             warped = cv2.warpPerspective(frame_bgr, matrix, (w, h))
        else:
             warped = frame_bgr

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        if exclusion_mask is not None:
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

    def calculate_projected_cte(self, cam_center, ppm, p_close, p_far, lookahead_m):
        vx, vy = cam_center
        vy_proj = vy - (lookahead_m * ppm) 

        dx = p_far[0] - p_close[0]
        dy = p_far[1] - p_close[1]
        path_len = math.hypot(dx, dy)
        
        if path_len < 1e-6:
            return 0.0, vy_proj 
            
        proj_cross_prod = dx * (vy_proj - p_close[1]) - dy * (vx - p_close[0])
        proj_cte = -(proj_cross_prod / path_len) / ppm
        
        return proj_cte, vy_proj
    
    def process_vision_pipeline(self, warped_img, cam_center, ppm, exclusion_mask=None, lookahead_m=0.0, rho=4.0, render_debug=True):
        debug_frame = warped_img.copy() if render_debug else None
        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        
        if exclusion_mask is not None:
            gray = cv2.bitwise_and(gray, exclusion_mask)
        
        h, w = gray.shape
        roi_mask = np.zeros_like(gray)
        roi_points = np.array([
            [360, 0],                         
            [910, 0],                         
            [int(w * 0.51), 450], 
            [int(w * 0.49), 450], 
        ], dtype=np.int32)
        cv2.fillPoly(roi_mask, [roi_points], 255)
        gray = cv2.bitwise_and(gray, roi_mask)
        
        if render_debug:
            cv2.polylines(debug_frame, [roi_points], isClosed=True, color=(255, 0, 0), thickness=2)

        lower_thresh, upper_thresh = self.lane_thresholds
        mask_binary = cv2.inRange(gray, lower_thresh, upper_thresh)
        
        morphed = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, self.morph_kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, self.morph_kernel)
        
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if render_debug:
            cv2.drawContours(debug_frame, contours, -1, (0, 255, 0), 1)

        path_points = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 15 < area < 550:
                rect = cv2.minAreaRect(cnt)
                rect_w, rect_h = rect[1]
                if rect_w > 0 and rect_h > 0:
                    aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h)
                    if aspect_ratio >= 2.0:
                        center = _contour_center(cnt)
                        if center:
                            path_points.append(center)
        
        path_points.sort(key=lambda p: p[1], reverse=True)

        cte = 0.0
        proj_cte = 0.0
        alpha = 0.0
        he = 0.0
        has_lock = False

        if len(path_points) >= 2:
            p_close = path_points[0]
            p_far = path_points[1] 
            
            if len(path_points) >= 3:
                p_alpha_A = path_points[1] 
                p_alpha_B = path_points[2] 
            else:
                p_alpha_A = path_points[0] 
                p_alpha_B = path_points[1]
                
            cte_alpha, _ = self.calculate_projected_cte(cam_center, ppm, p_alpha_A, p_alpha_B, rho)
            
            clipped_cte_alpha = max(min(cte_alpha, rho - 1e-4), -rho + 1e-4)
            alpha = math.asin(clipped_cte_alpha / rho)
            
            vx, vy = cam_center
            dx = p_far[0] - p_close[0]
            dy = p_far[1] - p_close[1]
            
            path_angle = math.atan2(-dy, dx) 
            he = (math.pi / 2) - path_angle
            path_len_sq = dx*dx + dy*dy
            
            calc_cte, vy = self.calculate_projected_cte(cam_center, ppm, p_close, p_far, 0.0)
            calc_proj_cte, vy_proj = self.calculate_projected_cte(cam_center, ppm, p_close, p_far, lookahead_m)
            
            if self.last_valid_cte is not None and abs(calc_cte - self.last_valid_cte) > 0.8:
                has_lock = False
            else:
                cte = calc_cte
                proj_cte = calc_proj_cte
                self.last_valid_cte = cte
                has_lock = True
        
            if render_debug and has_lock:
                cv2.line(debug_frame, p_close, p_far, (0, 255, 255), 2)
                if path_len_sq > 1e-6:
                    t = ((vx - p_close[0]) * dx + (vy - p_close[1]) * dy) / path_len_sq
                    proj_x = int(p_close[0] + t * dx)
                    proj_y = int(p_close[1] + t * dy)
                    cv2.line(debug_frame, (int(vx), int(vy)), (proj_x, proj_y), (0, 0, 255), 2)
                    cv2.circle(debug_frame, (proj_x, proj_y), 4, (0, 0, 255), -1)

                    t_proj = ((vx - p_close[0]) * dx + (vy_proj - p_close[1]) * dy) / path_len_sq
                    proj_x2 = int(p_close[0] + t_proj * dx)
                    proj_y2 = int(p_close[1] + t_proj * dy)
                    cv2.line(debug_frame, (int(vx), int(vy_proj)), (proj_x2, proj_y2), (255, 0, 255), 2)
                    cv2.circle(debug_frame, (int(vx), int(vy_proj)), 4, (255, 0, 255), -1)

                cv2.putText(debug_frame, f"CTE: {cte:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(debug_frame, f"Proj CTE: {proj_cte:.2f}m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(debug_frame, f"Alpha: {math.degrees(alpha):.2f} deg", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return cte, proj_cte, alpha, he, has_lock, debug_frame

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
        
        required_ids = {0, 1, 2, 3}
        step_sec = self.timestep / 1000.0
        frame_counter = 0

        while self.driver.step() != -1:
            frame_counter += 1
            key = self.keyboard.getKey()
            manual_override = False
            
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

            raw_speed_kmh, raw_steer = self.getspeedsteer()
            noisy_speed_kmh = raw_speed_kmh + np.random.normal(0, SPEED_NOISE_STD)
            noisy_steer = raw_steer + np.random.normal(0, STEER_NOISE_STD)
            speed_mps = noisy_speed_kmh / 3.6
            
            if not math.isnan(raw_speed_kmh) and not math.isnan(raw_steer):
                dist = speed_mps * step_sec
                if abs(dist) > 0.00001:
                    odom_yaw += (dist * math.tan(-noisy_steer) / WHEEL_BASE)
                    odom_x += dist * math.cos(odom_yaw)
                    odom_y += dist * math.sin(odom_yaw)

            raw_image = self.camera.getImage()
            if raw_image:
                img_np = np.frombuffer(raw_image, np.uint8).reshape((self.cam_height, self.cam_width, 4))
                frame_bgr = img_np[:, :, :3].copy()
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                run_tag_detect = (not self.bev_calibrated) or (frame_counter % TAG_DETECT_INTERVAL == 0)
                
                if run_tag_detect:
                    tags = self.at_detector.detect(gray, estimate_tag_pose=False, camera_params=self.camera_params, tag_size=self.tag_size_meters)
                    self.cached_tags = tags
                else:
                    tags = self.cached_tags

                tag_centers = {tag.tag_id: tag.center for tag in tags}
                h, w = frame_bgr.shape[:2]
                
                MASK_EXPANSION = 1.6
                if tags:
                    for tag in tags:
                        center = tag.center
                        corners = tag.corners
                        expanded_corners = center + (corners - center) * MASK_EXPANSION
                        pts = expanded_corners.astype(np.int32).reshape((-1, 1, 2))
                        self.active_tag_polys.append([pts, TAG_MASK_TTL_FRAMES])

                tag_mask = None
                if self.active_tag_polys:
                    tag_mask = np.ones((h, w), dtype=np.uint8) * 255
                    next_masks = []
                    for pts, ttl in self.active_tag_polys:
                        cv2.fillPoly(tag_mask, [pts], 0)
                        if ttl - 1 > 0:
                            next_masks.append([pts, ttl - 1])
                    self.active_tag_polys = next_masks

                should_run_calibration = required_ids.issubset(tag_centers.keys())

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

                if self.bev_calibrated and warped is not None:
                    warped_mask = cv2.warpPerspective(tag_mask, matrix, (w, h)) if tag_mask is not None else None
                    
                    render_this_frame = SHOW_GUI and (frame_counter % GUI_SKIP_INTERVAL == 0)
                    
                    cte, proj_cte, alpha, he, has_lock, debug_img = self.process_vision_pipeline(
                        warped, cam_center, ppm, warped_mask, lookahead_m=POSE_LOOKAHEAD_M, rho=4.0, render_debug=render_this_frame
                    )
                    
                    vis_success, vis_trans, vis_rot, theta_r = self.pose_estimator.update(
                        tags, self.camera_matrix, None, 
                        speed_mps, noisy_steer, step_sec,
                        cte=proj_cte, he=he, has_lane_lock=has_lock, lookahead_dist=POSE_LOOKAHEAD_M
                    )

                    if has_lock and not manual_override:
                        speed_kmh = StanleyController.calculate_velocity(cte, he, CRUISING_SPEED)
                        cmd_speed_mps = speed_kmh / 3.6
                        
                        # Added the kb parameter to match MATLAB logic
                        steer = BarrierLyapunovController.calculate_steering(
                            alpha=alpha,
                            heading_error=he, 
                            speed_mps=cmd_speed_mps, 
                            rho=4.0,              
                            alpha_bar_deg=33.0,   
                            wheelbase=WHEEL_BASE,
                            kb=1.0  
                        )
                        
                        self.current_steering = steer
                        self.current_speed = speed_kmh

                    if render_this_frame and self.bev_center is not None and debug_img is not None:
                        cx, cy = int(self.bev_center[0]), int(self.bev_center[1])
                        line_length = self.current_speed  
                        end_x = int(cx - line_length * math.sin(self.current_steering))
                        end_y = int(cy - line_length * math.cos(self.current_steering))
                        
                        cv2.line(debug_img, (cx, cy), (end_x, end_y), (0, 0, 255), 3)
                        cv2.circle(debug_img, (cx, cy), 5, (255, 0, 0), -1)
                        cv2.circle(debug_img, (end_x, end_y), 5, (0, 0, 255), -1)
                        cv2.putText(debug_img, f"Steer: {self.current_steering:.3f}", (10, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Ensure VideoWriter is setup and writing the frame
                        if self.video_writer is None:
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            vh, vw = debug_img.shape[:2]
                            # Auto-calculate recording FPS based on Webots timestep and rendering skips
                            fps = float(1000.0 / (self.timestep * GUI_SKIP_INTERVAL))
                            self.video_writer = cv2.VideoWriter('bev_driver_output.mp4', fourcc, fps, (vw, vh))
                        
                        self.video_writer.write(debug_img)

                    track_pos_est = (None, None, None)
                    gps_pos = None
                    car_rot = None
                    
                    if self.gps and self.imu and has_lock:
                        gps_pos = self.gps.getValues()
                        rpy = self.imu.getRollPitchYaw()
                        car_rot = np.degrees(rpy)
                        
                        car_yaw = rpy[2]
                        car_x, car_y, car_z = gps_pos
                        track_heading = car_yaw - he
                        
                        total_offset = self.pose_estimator.cam_offset_x + POSE_LOOKAHEAD_M
                        proj_car_x = car_x + total_offset * math.cos(car_yaw)
                        proj_car_y = car_y - total_offset * math.sin(car_yaw)
                        
                        t_x = proj_car_x - proj_cte * math.cos(track_heading - math.pi/2)
                        t_y = proj_car_y + proj_cte * math.sin(track_heading - math.pi/2)
                        
                        track_pos_est = (t_x, t_y, car_z)

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
                        theta_r=theta_r,
                        tag_detected=(len(tags) > 0)
                    )

                    if render_this_frame and debug_img is not None:
                        cv2.imshow("BEV Driver", debug_img)
                        cv2.imshow("Raw Camera", frame_bgr)
                        cv2.waitKey(1)
                else:
                    if render_this_frame:
                        cv2.putText(frame_bgr, "Waiting for Tags 0,1,2,3...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        cv2.imshow("BEV Driver", frame_bgr)
                        cv2.waitKey(1)

            self.driver.setCruisingSpeed(self.current_speed)
            self.driver.setSteeringAngle(self.current_steering)

        # Release video file handle on quit
        if self.video_writer is not None:
            self.video_writer.release()
            
        self.logger.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = WebotsLaneFollower()
    controller.run()