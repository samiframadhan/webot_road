import re
import yaml
import numpy as np
import os

# --- Configuration ---
INPUT_FILE = 'createroad.wbt'
OUTPUT_FILE = 'output.yaml'

def get_rotation_matrix(axis, theta):
    """
    Computes the 4x4 rotation matrix from an axis-angle representation using Rodrigues' formula.
    """
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        return np.eye(4)
    
    axis = axis / axis_norm
    
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
        [0, 0, 0, 1]
    ])

def get_translation_matrix(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T

def get_axis_angle_from_matrix(R):
    """
    Extracts the rotation axis (x, y, z) and angle (theta) from a 3x3 or 4x4 rotation matrix.
    Returns: [rx, ry, rz, theta]
    """
    # Ensure we are looking at the 3x3 rotation component
    R33 = R[:3, :3]
    
    # Calculate the angle using the trace of the matrix
    # Trace = 1 + 2cos(theta) => theta = arccos((Trace - 1) / 2)
    trace = np.trace(R33)
    # Clip to avoid numerical errors going slightly outside [-1, 1]
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    
    if angle < 1e-6:
        # If angle is near 0, the axis is arbitrary. standard is (0, 0, 1)
        return [0.0, 0.0, 1.0, 0.0]
    
    # Calculate the axis using the off-diagonal differences
    rx = R33[2, 1] - R33[1, 2]
    ry = R33[0, 2] - R33[2, 0]
    rz = R33[1, 0] - R33[0, 1]
    
    axis = np.array([rx, ry, rz])
    norm = np.linalg.norm(axis)
    
    if norm < 1e-6:
        # Fallback for singularities
        return [0.0, 0.0, 1.0, 0.0]
        
    axis = axis / norm
    
    return [float(axis[0]), float(axis[1]), float(axis[2]), float(angle)]

class WbtParser:
    def __init__(self, content):
        # Pre-processing: Pad brackets to make tokenization easier
        content = content.replace('{', ' { ').replace('}', ' } ')
        content = content.replace('[', ' [ ').replace(']', ' ] ')
        
        # Regex to capture strings with quotes, or regular tokens
        # This prevents breaking file paths inside quotes
        self.tokens = re.findall(r'"[^"]*"|[^\s\[\]{}]+|[\[\]{}]', content)
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self):
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            self.pos += 1
            return token
        return None

    def parse_block(self):
        """
        Parses a generic block (Solid, Shape, etc.) inside { ... }
        Returns a dictionary of properties found.
        """
        props = {
            'children': [], 
            'translation': [0, 0, 0], 
            'rotation': [0, 0, 1, 0], 
            'texture_id': None
        }
        
        while self.peek() != '}':
            key = self.consume()
            
            if key == 'translation':
                props['translation'] = [float(self.consume()), float(self.consume()), float(self.consume())]
            
            elif key == 'rotation':
                # Axis (x,y,z) and Angle (rad)
                props['rotation'] = [float(self.consume()), float(self.consume()), float(self.consume()), float(self.consume())]
            
            elif key == 'children':
                self.consume() # '['
                while self.peek() != ']':
                    if self.peek() == 'Solid':
                        self.consume() # 'Solid'
                        self.consume() # '{'
                        child_node = self.parse_block()
                        props['children'].append(child_node)
                        # The '}' is consumed by parse_block
                    elif self.peek() == 'Shape':
                        # We parse Shape to find texture IDs
                        self.consume() # 'Shape'
                        self.consume() # '{'
                        shape_props = self.parse_block()
                        if shape_props['texture_id'] is not None:
                            props['texture_id'] = shape_props['texture_id']
                    else:
                        # Skip other nodes (Transform, Group, etc) generic parsing
                        if self.peek() == '{':
                            self.consume()
                            self.skip_block()
                        else:
                            self.consume()
                self.consume() # ']'
            
            elif key == 'url':
                # Can be: url "string" OR url [ "string" ]
                next_token = self.consume()
                url_str = ""
                if next_token == '[':
                    url_str = self.consume()
                    self.consume() # ']'
                else:
                    url_str = next_token
                
                # Extract ID: tag36_11_00038_fixed.png -> 00038 -> 38 (int)
                match = re.search(r'(\d+)_fixed', url_str)
                if match:
                    props['texture_id'] = int(match.group(1)) # Convert to Integer
            
            elif key == '{':
                # Nested block (Appearance, Geometry, etc.)
                nested_props = self.parse_block()
                if nested_props['texture_id'] is not None:
                    props['texture_id'] = nested_props['texture_id']
            
            elif self.peek() is None:
                break
            
            # Ignore other keys (name, metalness, etc)
        
        self.consume() # '}'
        return props

    def skip_block(self):
        """Helper to skip blocks we don't care about"""
        count = 1
        while count > 0:
            token = self.consume()
            if token == '{': count += 1
            if token == '}': count -= 1

    def parse(self):
        roots = []
        while self.pos < len(self.tokens):
            if self.peek() == 'Solid':
                self.consume()
                self.consume() # '{'
                roots.append(self.parse_block())
            else:
                self.consume()
        return roots

def compute_global_transforms(nodes, parent_transform=np.eye(4), results=None):
    if results is None:
        results = {}

    for node in nodes:
        # 1. Local Transform
        tx, ty, tz = node['translation']
        rx, ry, rz, angle = node['rotation']
        
        T_trans = get_translation_matrix(tx, ty, tz)
        T_rot = get_rotation_matrix(np.array([rx, ry, rz]), angle)
        
        local_transform = T_trans @ T_rot
        
        # 2. Global Transform
        global_transform = parent_transform @ local_transform
        
        # 3. Store if ID exists
        if node['texture_id'] is not None:
            # Extract position
            x_glob, y_glob, z_glob = global_transform[:3, 3]
            
            # Extract global rotation as axis-angle
            # This replaces the previous yaw-only calculation
            axis_angle = get_axis_angle_from_matrix(global_transform)
            
            # Format: [Pos X, Pos Y, Pos Z, Axis X, Axis Y, Axis Z, Angle]
            results[node['texture_id']] = [
                float(x_glob), float(y_glob), float(z_glob),
                axis_angle[0], axis_angle[1], axis_angle[2], axis_angle[3]
            ]
        
        # 4. Recurse
        compute_global_transforms(node['children'], global_transform, results)

    return results

# --- Main Execution ---

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
    else:
        print(f"Reading {INPUT_FILE}...")
        with open(INPUT_FILE, 'r') as f:
            content = f.read()

        parser = WbtParser(content)
        root_nodes = parser.parse()
        
        print("Computing transforms...")
        final_data = {}
        # We need to flatten the list of roots into a single dict
        for root in root_nodes:
             # Process each top-level Solid
             # Note: We pass a list containing the single root node to the recursive function
             compute_global_transforms([root], np.eye(4), final_data)
        
        print(f"Found {len(final_data)} tagged solids.")
        
        # Sort by ID (Integer) for clean output
        sorted_data = {k: final_data[k] for k in sorted(final_data)}
        
        with open(OUTPUT_FILE, 'w') as f:
            # default_flow_style=None allows lists to be inline [x, y, z...] while keys are block style
            yaml.dump(sorted_data, f, default_flow_style=None)
            
        print(f"Successfully wrote to {OUTPUT_FILE}")
