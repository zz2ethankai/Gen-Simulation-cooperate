#!/usr/bin/env python3
"""Fix remaining issues with facing-target constraint."""

import json, math, yaml, os
from pathlib import Path

ROBOT_POLY = [
    ( 0.46,  0.24), ( 0.42,  0.29), (-0.32,  0.29),
    (-0.36,  0.24), (-0.36, -0.24), (-0.32, -0.29),
    ( 0.42, -0.29), ( 0.46, -0.24)
]

def polygon_at(nx, ny, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(nx + x*c - y*s, ny + x*s + y*c) for (x, y) in ROBOT_POLY]

def sat_project(poly, axis):
    dots = [p[0]*axis[0] + p[1]*axis[1] for p in poly]
    return min(dots), max(dots)

def polygons_intersect(poly_a, poly_b):
    for poly in [poly_a, poly_b]:
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1)%n]
            axis = (-(y2-y1), x2-x1)
            length = math.sqrt(axis[0]**2 + axis[1]**2)
            if length < 1e-9: continue
            axis = (axis[0]/length, axis[1]/length)
            min_a, max_a = sat_project(poly_a, axis)
            min_b, max_b = sat_project(poly_b, axis)
            if max_a < min_b or max_b < min_a:
                return False
    return True

def arm_reachable_facing(arm, nx, ny, yaw, tx, ty):
    if arm == "left":
        ax = nx + 0.368*math.cos(yaw) - 0.306*math.sin(yaw)
        ay = ny + 0.368*math.sin(yaw) + 0.306*math.cos(yaw)
    else:
        ax = nx + 0.368*math.cos(yaw) + 0.306*math.sin(yaw)
        ay = ny + 0.368*math.sin(yaw) - 0.306*math.cos(yaw)
    return math.hypot(ax - tx, ay - ty) <= 1.05

def robot_inside_room(rpoly, room_bounds):
    min_x = min(p[0] for p in rpoly)
    max_x = max(p[0] for p in rpoly)
    min_y = min(p[1] for p in rpoly)
    max_y = max(p[1] for p in rpoly)
    return (min_x >= room_bounds[0] + 0.10 and max_x <= room_bounds[1] - 0.10 and
            min_y >= room_bounds[2] + 0.10 and max_y <= room_bounds[3] - 0.10)

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / (dx*dx + dy*dy)))
    return math.hypot(px - (x1 + t*dx), py - (y1 + t*dy))

def min_dist_to_obstacles(rpoly, obstacles):
    min_d = float('inf')
    for o in obstacles:
        for p in rpoly:
            for i in range(len(o['poly'])):
                x1, y1 = o['poly'][i]
                x2, y2 = o['poly'][(i+1)%len(o['poly'])]
                dx, dy = x2-x1, y2-y1
                if dx == 0 and dy == 0:
                    d = math.hypot(p[0]-x1, p[1]-y1)
                else:
                    t = max(0, min(1, ((p[0]-x1)*dx + (p[1]-y1)*dy) / (dx*dx + dy*dy)))
                    d = math.hypot(p[0]-(x1+t*dx), p[1]-(y1+t*dy))
                if d < min_d:
                    min_d = d
    return min_d

def load_json(path):
    with open(path) as f:
        return json.load(f)

def metadata_size_xy(meta):
    layout = meta.get('layout_pose', {})
    size = layout.get('size_xyz_m')
    if isinstance(size, list) and len(size) >= 2:
        return float(size[0]), float(size[2])
    geom = meta.get('geometry_alignment', {})
    layout_size = geom.get('layout_size_xyz_m')
    if isinstance(layout_size, list) and len(layout_size) >= 2:
        return float(layout_size[0]), float(layout_size[2])
    return None

def fixture_size_xy(fixture, arena_path, task_root):
    size = fixture.get('size')
    if isinstance(size, list) and len(size) >= 2:
        return float(size[0]), float(size[1])
    
    source_metadata = fixture.get('source_metadata')
    candidate_paths = []
    if source_metadata:
        raw = Path(str(source_metadata))
        candidate_paths.append(raw if raw.is_absolute() else arena_path.parent / raw)
        if 'fixtures' in raw.parts:
            fixture_index = raw.parts.index('fixtures')
            candidate_paths.append(task_root / Path(*raw.parts[fixture_index:]))
    
    path = fixture.get('path')
    if path:
        raw = Path(str(path))
        asset_path = raw if raw.is_absolute() else arena_path.parent / raw
        candidate_paths.append(asset_path.parent / 'metadata.json')
        if 'fixtures' in raw.parts:
            fixture_index = raw.parts.index('fixtures')
            candidate_paths.append(task_root / Path(*raw.parts[fixture_index:]).parent / 'metadata.json')
    
    for candidate in candidate_paths:
        if candidate.is_file():
            try:
                meta = load_json(candidate)
                size_xy = metadata_size_xy(meta)
                if size_xy is not None:
                    scale = fixture.get('scale') or [1.0, 1.0, 1.0]
                    sx = float(scale[0]) if len(scale) > 0 else 1.0
                    sy = float(scale[2]) if len(scale) > 2 else 1.0
                    return size_xy[0] * sx, size_xy[1] * sy
            except (json.JSONDecodeError, FileNotFoundError):
                continue
    
    return None

def load_obstacles_from_arena(arena_path, task_root):
    with open(arena_path) as f:
        arena = yaml.safe_load(f)
    
    obstacles = []
    for fixture in arena.get('fixtures', []):
        if fixture.get('name') == 'floor':
            continue
        if fixture.get('collision_enabled') is False:
            continue
        
        role = str(fixture.get('role', ''))
        name = str(fixture.get('name', ''))
        translation = fixture.get('translation')
        if not isinstance(translation, list) or len(translation) < 2:
            continue
        
        if role == 'wall' or name.startswith('wall_'):
            continue
        
        size_xy = fixture_size_xy(fixture, arena_path, task_root)
        if size_xy is None:
            continue
        
        cx = float(translation[0])
        cy = float(translation[1])
        sx, sy = size_xy
        yaw = math.radians(float((fixture.get('euler') or [0.0, 0.0, 0.0])[2]))
        
        half_x, half_y = sx / 2.0, sy / 2.0
        corners_local = [(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)]
        c, s = math.cos(yaw), math.sin(yaw)
        poly = [(cx + x*c - y*s, cy + x*s + y*c) for (x, y) in corners_local]
        
        obstacles.append({
            'name': name,
            'poly': poly,
            'center_layout_xy': [cx, cy],
            'size_xy': [sx, sy]
        })
    
    return obstacles

def find_best_position_facing(overlay, tx, ty, obstacles, room_bounds):
    best = None
    best_margin = -1
    search_range = 1.37
    step = 0.02
    n_steps = int(search_range / step)
    
    for ix in range(-n_steps, n_steps+1):
        nx = tx + ix * step
        for iy in range(-n_steps, n_steps+1):
            ny = ty + iy * step
            dist_to_target = math.hypot(nx - tx, ny - ty)
            if dist_to_target < 0.15 or dist_to_target > search_range:
                continue
            
            yaw = math.atan2(ty - ny, tx - nx)
            rpoly = polygon_at(nx, ny, yaw)
            
            if not robot_inside_room(rpoly, room_bounds):
                continue
            
            collides = False
            for o in obstacles:
                if polygons_intersect(rpoly, o['poly']):
                    collides = True
                    break
            
            if collides:
                continue
            
            for arm in ["left", "right"]:
                if arm_reachable_facing(arm, nx, ny, yaw, tx, ty):
                    margin = min_dist_to_obstacles(rpoly, obstacles)
                    if margin > best_margin:
                        best = (nx, ny, yaw, arm, margin)
                        best_margin = margin
                    break
    
    if best:
        nx, ny, yaw, arm, margin = best
        best_fine = best
        for dx in [-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06]:
            for dy in [-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06]:
                nx2, ny2 = nx + dx, ny + dy
                dist_to_target = math.hypot(nx2 - tx, ny2 - ty)
                if dist_to_target < 0.15 or dist_to_target > search_range:
                    continue
                yaw2 = math.atan2(ty - ny2, tx - nx2)
                rpoly = polygon_at(nx2, ny2, yaw2)
                
                if not robot_inside_room(rpoly, room_bounds):
                    continue
                
                collides = False
                for o in obstacles:
                    if polygons_intersect(rpoly, o['poly']):
                        collides = True
                        break
                
                if collides:
                    continue
                
                for arm in ["left", "right"]:
                    if arm_reachable_facing(arm, nx2, ny2, yaw2, tx, ty):
                        margin = min_dist_to_obstacles(rpoly, obstacles)
                        if margin > best_fine[4]:
                            best_fine = (nx2, ny2, yaw2, arm, margin)
                        break
        return best_fine
    
    return None

with open('output/scene4_nav_skill_generation/scene4_nav_skill_generation_summary.json') as f:
    gen = json.load(f)

reports = {r['task']: r for r in gen['reports']}

# Fix 1: livingroom_toy_blocks_cleanup - move toy blocks to accessible location
print("="*60)
print("Fixing livingroom_toy_blocks_cleanup by moving toy blocks")

task = 'livingroom_toy_blocks_cleanup'
report = reports[task]
yaml_path = report['task_path']

with open(yaml_path) as f:
    task_data = yaml.safe_load(f)

task_entry = task_data['tasks'][0]

# Move toy blocks to accessible floor locations
new_block_positions = [
    (1.50, 1.20, 0.02),
    (1.70, 1.25, 0.02),
    (1.90, 1.15, 0.02),
]

block_idx = 0
for sr in task_entry['source_regions']:
    if 'toy_block' in sr['name'] and block_idx < len(new_block_positions):
        x, y, z = new_block_positions[block_idx]
        sr['center'] = [x, y]
        sr['support_surface_y'] = z
        block_idx += 1

# Also update regions section
region_idx = 0
for region in task_entry['regions']:
    if 'toy_block' in str(region.get('object', '')) and region_idx < len(new_block_positions):
        x, y, z = new_block_positions[region_idx]
        region['random_config']['pos_range'] = [[x, y, z], [x, y, z]]
        region_idx += 1

# Update generation summary pick target (use first block as representative)
report['pick_target_world_layout_xy'] = [1.50, 1.20]

# Now recompute nav points
overlay_path = f'output/scene4_nav_overlay_check/{task}_nav_points_overlay.json'
with open(overlay_path) as f:
    overlay = json.load(f)

arena_path = Path(report['arena_path'])
task_root = Path(report['task_path']).parent
obstacles = load_obstacles_from_arena(arena_path, task_root)
room_bounds = overlay['coordinate_frame']['room_bounds_xz']
floor_center = overlay['coordinate_frame']['floor_center_layout_xy']

tx, ty = 1.50, 1.20  # new pick target
best = find_best_position_facing(overlay, tx, ty, obstacles, room_bounds)
if best:
    nx, ny, yaw, arm, margin = best
    print(f"  nav_to_pick: ({nx:.3f}, {ny:.3f}) yaw={math.degrees(yaw):.1f}° margin={margin:.3f}m")
    pt = overlay['nav_points']['nav_to_pick']
    pt['world_x'] = nx
    pt['world_y'] = ny
    pt['local_x'] = nx - floor_center[0]
    pt['local_y'] = ny - floor_center[1]
    pt['yaw'] = yaw
    
    task_entry['positions']['nav_to_pick']['x'] = nx - floor_center[0]
    task_entry['positions']['nav_to_pick']['y'] = ny - floor_center[1]
    task_entry['positions']['nav_to_pick']['yaw'] = yaw
    
    overlay['obstacles'] = obstacles
    with open(overlay_path, 'w') as f:
        json.dump(overlay, f, indent=2)
    
    with open(yaml_path, 'w') as f:
        yaml.dump(task_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

# Fix 2: bedroom_hand_cream_to_organizer nav_to_place - search with finer grid
print("\n" + "="*60)
print("Fixing bedroom_hand_cream_to_organizer nav_to_place")

task = 'bedroom_hand_cream_to_organizer'
report = reports[task]
overlay_path = f'output/scene4_nav_overlay_check/{task}_nav_points_overlay.json'
with open(overlay_path) as f:
    overlay = json.load(f)

arena_path = Path(report['arena_path'])
task_root = Path(report['task_path']).parent
obstacles = load_obstacles_from_arena(arena_path, task_root)
room_bounds = overlay['coordinate_frame']['room_bounds_xz']
floor_center = overlay['coordinate_frame']['floor_center_layout_xy']

tx, ty = report['place_target_world_layout_xy']
print(f"  Place target: ({tx}, {ty})")

best = None
best_margin = -1
search_range = 1.37
step = 0.01
n_steps = int(search_range / step)

for ix in range(-n_steps, n_steps+1):
    nx = tx + ix * step
    for iy in range(-n_steps, n_steps+1):
        ny = ty + iy * step
        dist_to_target = math.hypot(nx - tx, ny - ty)
        if dist_to_target < 0.15 or dist_to_target > search_range:
            continue
        
        yaw = math.atan2(ty - ny, tx - nx)
        rpoly = polygon_at(nx, ny, yaw)
        
        if not robot_inside_room(rpoly, room_bounds):
            continue
        
        collides = False
        for o in obstacles:
            if polygons_intersect(rpoly, o['poly']):
                collides = True
                break
        
        if collides:
            continue
        
        for arm in ["left", "right"]:
            if arm_reachable_facing(arm, nx, ny, yaw, tx, ty):
                margin = min_dist_to_obstacles(rpoly, obstacles)
                if margin > best_margin:
                    best = (nx, ny, yaw, arm, margin)
                    best_margin = margin
                break

if best:
    nx, ny, yaw, arm, margin = best
    print(f"  nav_to_place: ({nx:.3f}, {ny:.3f}) yaw={math.degrees(yaw):.1f}° margin={margin:.3f}m")
    pt = overlay['nav_points']['nav_to_place']
    pt['world_x'] = nx
    pt['world_y'] = ny
    pt['local_x'] = nx - floor_center[0]
    pt['local_y'] = ny - floor_center[1]
    pt['yaw'] = yaw
    
    overlay['obstacles'] = obstacles
    with open(overlay_path, 'w') as f:
        json.dump(overlay, f, indent=2)
    
    yaml_path = report['task_path']
    with open(yaml_path) as f:
        task_data = yaml.safe_load(f)
    task_entry = task_data['tasks'][0]
    task_entry['positions']['nav_to_place']['x'] = nx - floor_center[0]
    task_entry['positions']['nav_to_place']['y'] = ny - floor_center[1]
    task_entry['positions']['nav_to_place']['yaw'] = yaw
    with open(yaml_path, 'w') as f:
        yaml.dump(task_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

# Save updated generation summary
with open('output/scene4_nav_skill_generation/scene4_nav_skill_generation_summary.json', 'w') as f:
    json.dump(gen, f, indent=2)

print("\nDone!")
