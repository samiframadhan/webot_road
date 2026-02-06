import cv2
import numpy as np
import argparse
import yaml
import math
from dt_apriltags import Detector

# Webots Imports
from vehicle import Driver
from controller import Keyboard

# --- Configuration ---
TAG_SIZE_METERS = 0.30
TAG_DIST_METERS = 1.20
VISUALIZE = True

# Control Constants
MAX_SPEED = 50.0
MAX_STEERING = 0.5
SPEED_STEP = 2.0
STEER_STEP = 0.02

class WebotsBEVDriver:
    def __init__(self, calibration_file):
        # 1. Initialize Webots Driver
        self.driver = Driver()
        self.timestep = int(self.driver.getBasicTimeStep())
        if self.timestep == 0:
            self.timestep = 32

        # 2. Initialize Camera
        # Ensure your robot's camera device is named "camera" in the scene tree
        self.camera = self.driver.getDevice("camera")
        if self.camera:
            self.camera.enable(self.timestep)
            self.cam_width = self.camera.getWidth()
            self.cam_height = self.camera.getHeight()
            self.cam_fov = self.camera.getFov()
            print(f"Webots Camera initialized: {self.cam_width}x{self.cam_height}, FOV: {self.cam_fov:.2f}")
        else:
            print("Error: Device 'camera' not found on robot.")
            self.cam_width = 640
            self.cam_height = 480
            self.cam_fov = 1.0

        # 3. Initialize Keyboard
        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)

        # 4. State & Calibration
        self.current_speed = 0.0
        self.current_steering = 0.0
        self.latest_correction = None
        self._load_calibration(calibration_file)

        # 5. Initialize AprilTag Detector
        self.detector = Detector(families="tag36h11",
                                 nthreads=2,
                                 quad_decimate=1.0,
                                 quad_sigma=0.0,
                                 refine_edges=1,
                                 decode_sharpening=0.25,
                                 debug=0)

    def _load_calibration(self, filename):
        """
        Loads calibration from file, or calculates ideal pinhole model 
        from Webots camera specs if file is missing.
        """
        try:
            fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                raise FileNotFoundError("File not open")
            
            self.K = fs.getNode("camera_matrix").mat()
            self.dist = fs.getNode("dist_coeffs").mat()
            print(f"Loaded calibration from {filename}")
        except Exception as e:
            print(f"Calibration file not found or invalid ({e}). Using Webots ideal model.")
            # Calculate K from Webots FOV
            # f = w / (2 * tan(fov/2))
            f_x = self.cam_width / (2 * math.tan(self.cam_fov / 2))
            f_y = f_x # Assume square pixels
            c_x = self.cam_width / 2
            c_y = self.cam_height / 2
            
            self.K = np.array([[f_x, 0, c_x],
                               [0, f_y, c_y],
                               [0, 0, 1]], dtype=np.float32)
            self.dist = np.zeros(5) # Ideal camera has no distortion

        self.inv_K = np.linalg.inv(self.K)

    # --- Math Helpers (Same as original) ---
    def get_ground_plane_equation(self, rvec, tvec):
        R, _ = cv2.Rodrigues(rvec)
        normal = R @ np.array([0, 0, 1]).reshape(3, 1)
        point_on_plane = tvec.reshape(3, 1)
        return normal, point_on_plane

    def ray_plane_intersection(self, pixel, normal, plane_point):
        u, v = pixel
        ray = self.inv_K @ np.array([u, v, 1.0]).reshape(3, 1)
        numerator = np.dot(normal.T, plane_point)
        denominator = np.dot(normal.T, ray)
        if abs(denominator) < 1e-6: return None
        lam = numerator / denominator
        return ray * lam

    def process_bev(self, img):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        cam_params = (self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2])
        tags = self.detector.detect(gray, estimate_tag_pose=False, camera_params=cam_params, tag_size=TAG_SIZE_METERS)
        
        tag_centers = {tag.tag_id: tag.center for tag in tags}

        # Check for tags 1-4 (IDs: 4, 7, 19, 16)
        target_ids = [4, 7, 19, 16]
        if not all(tid in tag_centers for tid in target_ids):
            return img

        # --- Define Rectangular Object Points ---
        tag_w = 1.5
        tag_h = 1.0
        
        obj_points = np.array([
            [-tag_w/2,  tag_h/2, 0], # TL
            [ tag_w/2,  tag_h/2, 0], # TR
            [ tag_w/2, -tag_h/2, 0], # BR
            [-tag_w/2, -tag_h/2, 0]  # BL
        ], dtype=np.float32)

        img_points = np.array([tag_centers[4], tag_centers[7], tag_centers[19], tag_centers[16]], dtype=np.float32)

        ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.K, self.dist)
        normal_cam, point_cam = self.get_ground_plane_equation(rvec, tvec)

        # --- Virtual Camera Setup ---
        cam_forward = np.array([0, 0, 1.0]).reshape(3, 1)
        proj_forward = cam_forward - (np.dot(cam_forward.T, normal_cam)) * normal_cam
        bev_up = proj_forward / (np.linalg.norm(proj_forward) + 1e-6)
        bev_right = np.cross(bev_up.flatten(), normal_cam.flatten()).reshape(3, 1)

        center_ground = self.ray_plane_intersection((w/2, h/2), normal_cam, point_cam)
        if center_ground is None: return img

        dist_to_center = np.linalg.norm(center_ground)
        focal_length = (self.K[0,0] + self.K[1,1]) / 2
        base_pixels_per_meter = focal_length / dist_to_center
        ZOOM_FACTOR = 0.2 
        pixels_per_meter = base_pixels_per_meter * ZOOM_FACTOR
        
        # Calculate Camera Position Logic
        cam_height = np.dot(-point_cam.T, normal_cam)
        nadir_3d = normal_cam * cam_height 
        diff_vec = nadir_3d - center_ground
        x_rel_meters = np.dot(diff_vec.flatten(), bev_right.flatten())
        y_rel_meters = np.dot(diff_vec.flatten(), bev_up.flatten())
        cam_pixel_x = (w / 2) + (x_rel_meters * pixels_per_meter)
        cam_pixel_y = (h / 2) - (y_rel_meters * pixels_per_meter)

        # Warp Generation
        w_m = w / pixels_per_meter
        h_m = h / pixels_per_meter
        p_tl = center_ground - (bev_right * w_m/2) + (bev_up * h_m/2)
        p_tr = center_ground + (bev_right * w_m/2) + (bev_up * h_m/2)
        p_br = center_ground + (bev_right * w_m/2) - (bev_up * h_m/2)
        p_bl = center_ground - (bev_right * w_m/2) - (bev_up * h_m/2)
        
        pts_3d = np.array([p_tl, p_tr, p_br, p_bl]).reshape(4, 3)
        pts_src, _ = cv2.projectPoints(pts_3d, np.zeros(3), np.zeros(3), self.K, self.dist)
        pts_dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        
        matrix = cv2.getPerspectiveTransform(pts_src.reshape(4, 2).astype(np.float32), pts_dst)
        warped = cv2.warpPerspective(img, matrix, (w, h))
        
        self.latest_correction = (matrix, cam_pixel_x, cam_pixel_y, pixels_per_meter, cam_height)

        # Visualization
        cx, cy = int(cam_pixel_x), int(cam_pixel_y)
        if -2000 < cx < 2000 and -2000 < cy < 2000:
            cv2.circle(warped, (cx, cy), 10, (0, 255, 255), -1)
            cv2.putText(warped, "CAM", (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.line(warped, (cx, cy), (int(w/2), int(h/2)), (0, 255, 255), 1)

        return warped

    def update_control(self):
        """Reads keyboard and updates driver"""
        key = self.keyboard.getKey()
        
        # Simple non-blocking key check (Webots returns -1 if no key)
        while key > 0:
            if key == Keyboard.UP:
                self.current_speed += SPEED_STEP
            elif key == Keyboard.DOWN:
                self.current_speed -= SPEED_STEP
            elif key == Keyboard.LEFT:
                self.current_steering -= STEER_STEP
            elif key == Keyboard.RIGHT:
                self.current_steering += STEER_STEP
            elif key == ord(' '): # Brake
                self.current_speed = 0
                self.current_steering = 0
            
            key = self.keyboard.getKey()

        # Clamp values
        self.current_speed = max(min(self.current_speed, MAX_SPEED), -MAX_SPEED)
        self.current_steering = max(min(self.current_steering, MAX_STEERING), -MAX_STEERING)
        
        # Send to Webots
        self.driver.setCruisingSpeed(self.current_speed)
        self.driver.setSteeringAngle(self.current_steering)

    def run(self):
        print("Starting Simulation. Click the 3D view to focus, then use Arrow Keys.")
        
        try:
            # Main Webots Loop
            while self.driver.step() != -1:
                # 1. Update Vehicle Physics
                self.update_control()

                # 2. Process Camera
                if self.camera:
                    raw_image = self.camera.getImage()
                    if raw_image:
                        # Webots returns raw bytes (BGRA), OpenCV needs np array
                        img = np.frombuffer(raw_image, np.uint8).reshape((self.cam_height, self.cam_width, 4))
                        
                        # Remove Alpha channel for OpenCV processing (BGRA -> BGR)
                        frame_bgr = img[:, :, :3].copy() 

                        # Run BEV
                        bev_frame = self.process_bev(frame_bgr)

                        if VISUALIZE:
                            # cv2.imshow("Original", frame_bgr)
                            # Source - https://stackoverflow.com/a/58298221
                            # Posted by David Mansolino, modified by community. See post 'Timeline' for change history
                            # Retrieved 2026-02-05, License - CC BY-SA 4.0
                        
                            # imageRef = display.imageNew(camera.getImage().tolist(), Display.RGB)
                            # display.imagePaste(imageRef, 0, 0)
    
                            cv2.imshow("Bird's Eye View", bev_frame)
                            cv2.waitKey(1)
        finally:
            self.cleanup()

    def cleanup(self):
        if self.latest_correction is not None:
            with open("latest_correction_matrix.yaml", "w") as f:
                yaml.dump({
                    "matrix": self.latest_correction[0].tolist(),
                    "camera_pixel_x": float(self.latest_correction[1]),
                    "camera_pixel_y": float(self.latest_correction[2]),
                    "pixels_per_meter": float(self.latest_correction[3]),
                    "camera_height_meters": float(self.latest_correction[4])
                }, f)
            print("Saved latest correction matrix.")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", type=str, default="calibration_webots.yaml")
    args = parser.parse_args()

    bev_driver = WebotsBEVDriver(args.calib)
    bev_driver.run()