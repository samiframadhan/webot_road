import re
import yaml
import numpy as np
import os

# --- Configuration ---
INPUT_FILE = 'createroad.wbt'
OUTPUT_FILE = 'output3.yaml'

def get_rotation_matrix(axis, theta):
    """Computes 4x4 rotation matrix from axis-angle."""
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0: return np.eye(4)
    axis = axis / axis_norm
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([
        [aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac), 0],
        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab), 0],
        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc, 0],
        [0, 0, 0, 1]
    ])

def get_translation_matrix(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T

def get_axis_angle_from_matrix(R):
    """Extracts axis-angle [rx, ry, rz, theta] from rotation matrix."""
    R33 = R[:3, :3]
    trace = np.trace(R33)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    if angle < 1e-6: return [0.0, 0.0, 1.0, 0.0]
    rx = R33[2, 1] - R33[1, 2]
    ry = R33[0, 2] - R33[2, 0]
    rz = R33[1, 0] - R33[0, 1]
    axis = np.array([rx, ry, rz])
    axis = axis / np.linalg.norm(axis)
    return [float(axis[0]), float(axis[1]), float(axis[2]), float(angle)]

class WbtParser:
    def __init__(self, content):
        # Remove comments (lines starting with # or content after #)
        lines = [line.split('#')[0] for line in content.split('\n')]
        content = ' '.join(lines)
        
        # Pad brackets for easier tokenization
        content = content.replace('{', ' { ').replace('}', ' } ')
        content = content.replace('[', ' [ ').replace(']', ' ] ')
        self.tokens = re.findall(r'"[^"]*"|[^\s\[\]{}]+|[\[\]{}]', content)
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self):
        if self.pos < len(self.tokens):
            t = self.tokens[self.pos]
            self.pos += 1
            return t
        return None

    def parse_block(self):
        props = {
            'children': [], 
            'translation': [0, 0, 0], 
            'rotation': [0, 0, 1, 0], 
            'texture_id': None
        }
        
        while self.peek() != '}':
            key = self.consume()
            
            if key == 'translation':
                props['translation'] = [float(self.consume()) for _ in range(3)]
            elif key == 'rotation':
                props['rotation'] = [float(self.consume()) for _ in range(4)]
            elif key == 'children':
                self.consume() # '['
                while self.peek() != ']':
                    # Handle nodes (Solid, Transform, etc.)
                    # If we see DEF, consume it and the name, then check node type
                    if self.peek() == 'DEF':
                        self.consume() # DEF
                        self.consume() # Name
                    
                    if self.peek() in ['Solid', 'Transform']:
                        self.consume() # Node type
                        self.consume() # '{'
                        child_node = self.parse_block()
                        props['children'].append(child_node)
                    elif self.peek() == 'Shape':
                        self.consume() # Shape
                        self.consume() # '{'
                        shape_props = self.parse_block()
                        if shape_props['texture_id'] is not None:
                            props['texture_id'] = shape_props['texture_id']
                    else:
                        # Skip other nodes
                        if self.peek() == '{':
                            self.consume()
                            self.skip_block()
                        else:
                            self.consume()
                self.consume() # ']'
            elif key == 'url':
                next_token = self.consume()
                url_str = self.consume() if next_token == '[' else next_token
                if next_token == '[': self.consume() # ']'
                
                # Extract ID: tag36_11_00038_fixed.png -> 38
                match = re.search(r'tag.*_(\d+)_fixed', url_str)
                if match:
                    props['texture_id'] = int(match.group(1))
            elif key == '{':
                nested = self.parse_block()
                if nested['texture_id'] is not None:
                    props['texture_id'] = nested['texture_id']
            elif self.peek() is None:
                break
        
        self.consume() # '}'
        return props

    def skip_block(self):
        count = 1
        while count > 0:
            t = self.consume()
            if t == '{': count += 1
            if t == '}': count -= 1

    def parse(self):
        roots = []
        while self.pos < len(self.tokens):
            if self.peek() == 'DEF':
                self.consume()
                self.consume()
            
            if self.peek() in ['Solid', 'Transform']:
                self.consume()
                self.consume() # '{'
                roots.append(self.parse_block())
            else:
                self.consume()
        return roots

def compute_global_transforms(nodes, parent_transform=np.eye(4), results=None):
    if results is None: results = {}
    for node in nodes:
        tx, ty, tz = node['translation']
        rx, ry, rz, angle = node['rotation']
        
        # Webots Transform Order: T * R
        T_trans = get_translation_matrix(tx, ty, tz)
        T_rot = get_rotation_matrix(np.array([rx, ry, rz]), angle)
        local_transform = T_trans @ T_rot
        
        # Chain: Global = Parent * Local
        global_transform = parent_transform @ local_transform
        
        if node['texture_id'] is not None:
            x, y, z = global_transform[:3, 3]
            axis_angle = get_axis_angle_from_matrix(global_transform)
            results[node['texture_id']] = [float(x), float(y), float(z), *axis_angle]
        
        # Recurse with the new global transform as the parent
        compute_global_transforms(node['children'], global_transform, results)
    return results

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
    else:
        print(f"Reading {INPUT_FILE}...")
        with open(INPUT_FILE, 'r') as f:
            content = f.read()

        parser = WbtParser(content)
        root_nodes = parser.parse()
        
        final_data = {}
        compute_global_transforms(root_nodes, np.eye(4), final_data)
        
        print(f"Found {len(final_data)} tagged objects.")
        sorted_data = {k: final_data[k] for k in sorted(final_data)}
        
        with open(OUTPUT_FILE, 'w') as f:
            yaml.dump(sorted_data, f, default_flow_style=None)
        print(f"Written to {OUTPUT_FILE}")