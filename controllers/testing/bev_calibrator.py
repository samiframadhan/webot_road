import cv2
import numpy as np
import math

class BEVCalibrator:
    def __init__(self, cam_width, cam_height, cam_fov):
        self.cam_width = cam_width
        self.cam_height = cam_height
        
        # Initialize Camera Matrix (Ideal Pinhole Model based on Webots FOV)
        self._init_camera_matrix(cam_fov)
        
        # State storage for the last known good calibration
        self.latest_matrix = None
        self.pixels_per_meter = 100.0 # Default fallback
        self.cam_offset_x = 0.0
        self.cam_offset_y = 0.0

    def _init_camera_matrix(self, fov):
        """Calculates K based on Webots Field of View."""
        f_x = self.cam_width / (2 * math.tan(fov / 2))
        f_y = f_x  # Assume square pixels
        c_x = self.cam_width / 2
        c_y = self.cam_height / 2
        
        self.K = np.array([[f_x, 0, c_x],
                           [0, f_y, c_y],
                           [0, 0, 1]], dtype=np.float32)
        self.dist = np.zeros(5) # No distortion in simulation
        self.inv_K = np.linalg.inv(self.K)

    def _get_ground_plane(self, rvec, tvec):
        R, _ = cv2.Rodrigues(rvec)
        normal = R @ np.array([0, 0, 1]).reshape(3, 1)
        point = tvec.reshape(3, 1)
        return normal, point

    def _ray_plane_intersection(self, pixel, normal, plane_point):
        u, v = pixel
        ray = self.inv_K @ np.array([u, v, 1.0]).reshape(3, 1)
        denom = np.dot(normal.T, ray)
        if abs(denom) < 1e-6: return None
        lam = np.dot(normal.T, plane_point) / denom
        return ray * lam

    def process(self, img, tag_centers):
        """
        Main pipeline:
        1. Checks if tags 0-3 are in tag_centers.
        2. If found -> Recalculate Matrix & Update State.
        3. If not found -> Use cached Matrix.
        4. Return warped image + metrics.
        """
        h, w = img.shape[:2]

        # Target IDs: 0 (TL), 1 (TR), 2 (BR), 3 (BL)
        target_ids = [0, 1, 2, 3] 
        tags_found = all(tid in tag_centers for tid in target_ids)

        if tags_found:
            # --- Calibration Logic (Only runs when tags are visible) ---
            tag_w = 1.5  # Physical width between tag centers (meters)
            tag_h = 1.0  # Physical length between tag centers (meters)
            
            # Object Points (Physical World)
            obj_points = np.array([
                [-tag_w/2,  tag_h/2, 0], # TL (0)
                [ tag_w/2,  tag_h/2, 0], # TR (1)
                [ tag_w/2, -tag_h/2, 0], # BR (2)
                [-tag_w/2, -tag_h/2, 0]  # BL (3)
            ], dtype=np.float32)

            # Image Points (Pixel Coords) - retrieve from passed dict
            img_points = np.array([
                tag_centers[0], 
                tag_centers[1], 
                tag_centers[2], 
                tag_centers[3]
            ], dtype=np.float32)

            # Solve PnP
            ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.K, self.dist)
            normal_cam, point_cam = self._get_ground_plane(rvec, tvec)

            # Virtual Camera Maths
            cam_forward = np.array([0, 0, 1.0]).reshape(3, 1)
            proj_forward = cam_forward - (np.dot(cam_forward.T, normal_cam)) * normal_cam
            bev_up = proj_forward / (np.linalg.norm(proj_forward) + 1e-6)
            bev_right = np.cross(bev_up.flatten(), normal_cam.flatten()).reshape(3, 1)

            center_ground = self._ray_plane_intersection((w/2, h/2), normal_cam, point_cam)
            
            if center_ground is not None:
                dist_to_center = np.linalg.norm(center_ground)
                focal_length = (self.K[0,0] + self.K[1,1]) / 2
                base_ppm = focal_length / dist_to_center
                
                # --- Configuration ---
                ZOOM_FACTOR = 0.2
                self.pixels_per_meter = base_ppm * ZOOM_FACTOR
                
                # Calculate Camera Pixel Position in BEV
                cam_height = np.dot(-point_cam.T, normal_cam)
                nadir_3d = normal_cam * cam_height 
                diff_vec = nadir_3d - center_ground
                
                # Calculate relative position in meters
                x_rel = np.dot(diff_vec.flatten(), bev_right.flatten())
                y_rel = np.dot(diff_vec.flatten(), bev_up.flatten())
                
                # Update State
                self.cam_offset_x = (w / 2) + (x_rel * self.pixels_per_meter)
                self.cam_offset_y = (h / 2) - (y_rel * self.pixels_per_meter)

                # Generate Perspective Matrix
                w_m = w / self.pixels_per_meter
                h_m = h / self.pixels_per_meter
                
                p_tl = center_ground - (bev_right * w_m/2) + (bev_up * h_m/2)
                p_tr = center_ground + (bev_right * w_m/2) + (bev_up * h_m/2)
                p_br = center_ground + (bev_right * w_m/2) - (bev_up * h_m/2)
                p_bl = center_ground - (bev_right * w_m/2) - (bev_up * h_m/2)
                
                pts_3d = np.array([p_tl, p_tr, p_br, p_bl]).reshape(4, 3)
                pts_src, _ = cv2.projectPoints(pts_3d, np.zeros(3), np.zeros(3), self.K, self.dist)
                pts_dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
                
                self.latest_matrix = cv2.getPerspectiveTransform(pts_src.reshape(4, 2).astype(np.float32), pts_dst)
                
                # Visual Feedback for Calibration Success
                cv2.putText(img, "CALIBRATION UPDATED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # --- Warping ---
        if self.latest_matrix is not None:
            warped = cv2.warpPerspective(img, self.latest_matrix, (w, h))
            is_calibrated = True
        else:
            warped = img.copy()
            cv2.putText(warped, "NO CALIBRATION", (int(w/2)-100, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            is_calibrated = False

        return warped, self.latest_matrix, (self.cam_offset_x, self.cam_offset_y), self.pixels_per_meter, is_calibrated
