import cv2
import numpy as np
import math
from dt_apriltags import Detector

# Webots Imports
from vehicle import Driver
from controller import Keyboard

# Local Imports
from bev_calibrator import BEVCalibrator

# --- Configuration ---
MAX_SPEED_WEBOTS = 50.0 
CRUISING_SPEED = 100.0
STANLEY_K = 1.0

# --- Helper Functions ---
def _contour_center(cnt):
    M = cv2.moments(cnt)
    if M["m00"] == 0: return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

class StanleyController:
    @staticmethod
    def calculate_steering(cross_track_error, heading_error, speed, k=0.8, epsilon=1e-5):
        denom = abs(speed*0.277778) + epsilon
        cross_track_term = math.atan2(k * -cross_track_error, denom)
        return heading_error + cross_track_term
    def calculate_velocity(cross_track_error, heading_error, max_speed=1.0):
        error_magnitude = abs(cross_track_error) + abs(heading_error)
        speed_factor = 1.0 / (1.0 + error_magnitude)
        desired_speed = max_speed * speed_factor
        return max(0.2, desired_speed)

class WebotsLaneFollower:
    def __init__(self):
        # 1. Initialize Webots Driver
        self.driver = Driver()
        self.timestep = int(self.driver.getBasicTimeStep())
        if self.timestep == 0: self.timestep = 32

        # 2. Initialize Camera
        self.camera = self.driver.getDevice("camera")
        if self.camera:
            self.camera.enable(self.timestep)
            self.cam_width = self.camera.getWidth()
            self.cam_height = self.camera.getHeight()
            fov = self.camera.getFov()
            print(f"Webots Camera: {self.cam_width}x{self.cam_height}, FOV: {fov:.2f}")
            
            self.bev = BEVCalibrator(self.cam_width, self.cam_height, fov)
            self.at_detector = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
            self.tag_size_meters = 0.30
            self.camera_params = (self.bev.K[0,0], self.bev.K[1,1], self.bev.K[0,2], self.bev.K[1,2])
        else:
            print("Error: Camera not found")

        # 3. Initialize Keyboard
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)

        # 4. Control State
        self.current_speed = 0.0
        self.current_steering = 0.0
        
        self.lane_thresholds = (190, 255) 
        self.calibrated = False

        # 5. BEV Persistence (New Additions)
        self.bev_calibrated = False
        self.M = None           # Perspective Matrix
        self.bev_center = None  # Camera center in BEV
        self.bev_ppm = None     # Pixels per meter

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
        else:
            print("Calibration Failed: No clear peaks found.")

    def process_vision_pipeline(self, warped_img, cam_center, ppm, exclusion_mask=None):
        """
        Detects lane lines using simple contour centers and visualizes CTE.
        """
        debug_frame = warped_img.copy()
        
        # --- 1. Color Filtering ---
        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        if exclusion_mask is not None:
            gray = cv2.bitwise_and(gray, exclusion_mask)

        lower_thresh, upper_thresh = self.lane_thresholds
        mask_binary = cv2.inRange(gray, lower_thresh, upper_thresh)
        
        kernel = np.ones((3,3), np.uint8)
        morphed = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
        
        # --- 2. Contours ---
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug_frame, contours, -1, (0, 255, 0), 1)
        
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < 550]

        # --- 3. Simple Path Estimation (No Pairing) ---
        path_points = []
        for cnt in valid_contours:
            center = _contour_center(cnt)
            if center:
                path_points.append(center)
        
        # Sort by Y (Closest to car first -> Higher Y value)
        path_points.sort(key=lambda p: p[1], reverse=True)

        cte = 0.0
        he = 0.0
        has_lock = False

        if len(path_points) >= 2:
            # Use closest two points for local path approximation
            p_close = path_points[0]
            p_far = path_points[1] 
            
            # Draw Path Line
            cv2.line(debug_frame, p_close, p_far, (0, 255, 255), 2)
            
            # --- Calculate Errors ---
            dx = p_far[0] - p_close[0]
            dy = p_far[1] - p_close[1]
            path_angle = math.atan2(-dy, dx) 
            he = (math.pi / 2) - path_angle

            # Cross Track Error
            vx, vy = cam_center
            
            if abs(dx) < 1e-5: # Vertical line
                cte_pixels = vx - p_close[0]
                # Visual: Horizontal line to path
                proj_pt = (p_close[0], int(vy))
            else:
                m = dy / dx
                c_val = p_close[1] - (m * p_close[0])
                cte_pixels = (m * vx - vy + c_val) / math.sqrt(m**2 + 1)
                
                # Visual: Calculate projection point for drawing CTE line
                # Line 1 (Path): y = mx + c
                # Line 2 (Perpendicular from Car): y - vy = (-1/m)(x - vx)
                epsilon = 1e-5
                m_perp = -1/(m+epsilon)
                c_perp = vy - (m_perp * vx)
                
                # Intersection
                inter_x = (c_perp - c_val) / (m - m_perp)
                inter_y = m * inter_x + c_val
                proj_pt = (int(inter_x), int(inter_y))
            
            cte = cte_pixels / ppm
            has_lock = True
            
            # --- Visualization: CTE ---
            # Draw line from Car Center to Projection Point on Path (Red)
            cv2.line(debug_frame, (int(vx), int(vy)), proj_pt, (0, 0, 255), 2)
            cv2.circle(debug_frame, proj_pt, 4, (0, 0, 255), -1)
            
            # Text Stats
            cv2.putText(debug_frame, f"CTE: {cte:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(debug_frame, f"HE: {math.degrees(he):.1f}deg", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return cte, he, has_lock, debug_frame

    def run(self):
        print("Starting Webots Lane Follower...")
        print("Controls: Arrows to move, Space to stop, 'C' to Calibrate.")
        
        while self.driver.step() != -1:
            key = self.keyboard.getKey()
            manual_override = False
            
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
            elif key == ord('C'):
                pass 

            raw_image = self.camera.getImage()
            if raw_image:
                img_np = np.frombuffer(raw_image, np.uint8).reshape((self.cam_height, self.cam_width, 4))
                frame_bgr = img_np[:, :, :3].copy()
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                tags = self.at_detector.detect(gray, estimate_tag_pose=False, camera_params=self.camera_params, tag_size=self.tag_size_meters)
                tag_centers = {tag.tag_id: tag.center for tag in tags}
                
                h, w = frame_bgr.shape[:2]
                tag_mask = np.ones((h, w), dtype=np.uint8) * 255
                MASK_EXPANSION = 1.4

                # Create Exclusion mask for tags
                if tags:
                    for tag in tags:
                        center = tag.center
                        corners = tag.corners
                        expanded_corners = center + (corners - center) * MASK_EXPANSION
                        pts = expanded_corners.astype(np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(tag_mask, [pts], 0)

                # --- BEV Calibration Logic (Modified) ---
                required_ids = {0, 1, 2, 3}
                present_ids = set(tag_centers.keys())
                
                # Check if we have all specific IDs to run a FRESH calibration
                should_run_calibration = required_ids.issubset(present_ids)

                warped = None
                matrix = None
                cam_center = None
                ppm = None
                
                if should_run_calibration:
                    # Run BEV process and update persistent data
                    warped, matrix, cam_center, ppm, is_valid = self.bev.process(frame_bgr, tag_centers)
                    if is_valid:
                        self.M = matrix
                        self.bev_center = cam_center
                        self.bev_ppm = ppm
                        self.bev_calibrated = True
                
                elif self.bev_calibrated:
                    # No tags, but we have old data. Manual Warp.
                    matrix = self.M
                    cam_center = self.bev_center
                    ppm = self.bev_ppm
                    warped = cv2.warpPerspective(frame_bgr, matrix, (w, h))

                # --- Pipeline Execution ---
                
                # Handle 'C' key calibration (needs matrix)
                if key == ord('C'):
                     self.calibrate_lane_thresholds(frame_bgr, tag_mask, matrix)

                if self.bev_calibrated and warped is not None:
                    # Warp the tag mask to match the BEV view
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
                        # cv2.putText(debug_img, "AUTO", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(debug_img, f"Steer angle: {math.degrees(steer):.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # --- Visualization: Steering Angle ---
                    vx, vy = cam_center
                    steering_len = 50
                    steer_vis_angle = -math.pi/2 + self.current_steering
                    
                    sx = int(vx + steering_len * math.cos(steer_vis_angle))
                    sy = int(vy + steering_len * math.sin(steer_vis_angle))
                    
                    cv2.arrowedLine(debug_img, (int(vx), int(vy)), (sx, sy), (255, 0, 255), 3)

                    cv2.imshow("BEV Driver", debug_img)
                else:
                    # Not calibrated yet, show raw feed with warning
                    cv2.putText(frame_bgr, "Waiting for Tags 0,1,2,3...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.imshow("BEV Driver", frame_bgr)
                    
                cv2.waitKey(1)

            self.driver.setCruisingSpeed(self.current_speed)
            self.driver.setSteeringAngle(self.current_steering)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = WebotsLaneFollower()
    controller.run()