import math
from PIL import Image, ImageDraw, ImageFilter

SIZE = 512
SCALE = 2  # Super-sample at 1024x1024 for ultra-smooth antialiasing
DIM = SIZE * SCALE
img = Image.new("RGBA", (DIM, DIM), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

cx, cy = DIM // 2, DIM // 2
r_outer = int(DIM * 0.46)

# 1. Subtle Outer Glow
for i in range(15):
    alpha = int(18 * (1.0 - i / 15.0))
    glow_r = r_outer + i * 4
    draw.ellipse([cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r],
                 outline=(0, 240, 255, alpha), width=3)

# 2. Main Background Squircle / Rounded Disk with Gradient
bg_mask = Image.new("L", (DIM, DIM), 0)
bg_mask_draw = ImageDraw.Draw(bg_mask)
bg_mask_draw.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], fill=255)

# Gradient fill for background
bg_gradient = Image.new("RGBA", (DIM, DIM), (11, 15, 25, 255))
bg_draw = ImageDraw.Draw(bg_gradient)
for y in range(DIM):
    ratio = y / DIM
    # From deep dark navy #0a0e17 to darker slate #161e2e
    r = int(10 + ratio * 12)
    g = int(14 + ratio * 16)
    b = int(23 + ratio * 28)
    bg_draw.line([(0, y), (DIM, y)], fill=(r, g, b, 255))

img.paste(bg_gradient, (0, 0), bg_mask)

# 3. Outer Ring Border (Neon Cyan / Electric Blue)
draw.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
             outline=(0, 229, 255, 240), width=SCALE * 4)
draw.ellipse([cx - r_outer + SCALE*4, cy - r_outer + SCALE*4, cx + r_outer - SCALE*4, cy + r_outer - SCALE*4],
             outline=(14, 165, 233, 120), width=SCALE * 2)

# 4. Subtle Radar / HUD Rings in Background
for rad in [int(DIM * 0.18), int(DIM * 0.28), int(DIM * 0.38)]:
    draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                 outline=(0, 240, 255, 30), width=SCALE * 1)

# Subtle crosshair ticks
draw.line([(cx - int(DIM * 0.40), cy), (cx - int(DIM * 0.12), cy)], fill=(0, 240, 255, 40), width=SCALE)
draw.line([(cx + int(DIM * 0.12), cy), (cx + int(DIM * 0.40), cy)], fill=(0, 240, 255, 40), width=SCALE)
draw.line([(cx, cy - int(DIM * 0.40)), (cx, cy - int(DIM * 0.12))], fill=(0, 240, 255, 40), width=SCALE)
draw.line([(cx, cy + int(DIM * 0.12)), (cx, cy + int(DIM * 0.40))], fill=(0, 240, 255, 40), width=SCALE)

# 5. Drone Arm Geometry (X shape)
arm_dist = int(DIM * 0.24)
rotor_radius = int(DIM * 0.13)
angles = [45, 135, 225, 315]  # FR, BR, BL, FL

# Draw carbon fiber arms
for ang in angles:
    rad = math.radians(ang)
    x_motor = cx + arm_dist * math.cos(rad)
    y_motor = cy + arm_dist * math.sin(rad)
    
    # Outer arm shadow / stroke
    draw.line([(cx, cy), (x_motor, y_motor)], fill=(15, 23, 42, 255), width=SCALE * 16)
    # Inner carbon arm
    draw.line([(cx, cy), (x_motor, y_motor)], fill=(51, 65, 85, 255), width=SCALE * 10)
    # Highlight accent line
    draw.line([(cx, cy), (x_motor, y_motor)], fill=(0, 229, 255, 180), width=SCALE * 2)

# 6. Rotor Disks (Spinning motion blur + Propellers)
for idx, ang in enumerate(angles):
    rad = math.radians(ang)
    x_m = cx + arm_dist * math.cos(rad)
    y_m = cy + arm_dist * math.sin(rad)
    
    # Spinning blur disc
    if idx in [0, 3]:  # Front rotors (Cyan glow)
        glow_col = (0, 229, 255, 45)
        ring_col = (0, 240, 255, 160)
        nav_col = (255, 100, 50, 255)  # Orange front nav light
    else:  # Rear rotors (Green / Teal glow)
        glow_col = (16, 185, 129, 45)
        ring_col = (52, 211, 153, 160)
        nav_col = (16, 185, 129, 255)  # Green rear nav light

    # Propeller disc outer ring
    draw.ellipse([x_m - rotor_radius, y_m - rotor_radius, x_m + rotor_radius, y_m + rotor_radius],
                 outline=ring_col, width=SCALE * 2)
    # Propeller disc inner concentric ring
    inner_r = int(rotor_radius * 0.6)
    draw.ellipse([x_m - inner_r, y_m - inner_r, x_m + inner_r, y_m + inner_r],
                 outline=(*ring_col[:3], 60), width=SCALE)

    # 2-blade Carbon Propellers
    prop_ang = ang + 25
    p_rad1 = math.radians(prop_ang)
    p_rad2 = p_rad1 + math.pi
    p1 = (x_m + (rotor_radius - 6*SCALE) * math.cos(p_rad1), y_m + (rotor_radius - 6*SCALE) * math.sin(p_rad1))
    p2 = (x_m + (rotor_radius - 6*SCALE) * math.cos(p_rad2), y_m + (rotor_radius - 6*SCALE) * math.sin(p_rad2))
    
    draw.line([p1, p2], fill=(226, 232, 240, 230), width=SCALE * 5)
    draw.line([p1, p2], fill=(0, 229, 255, 200), width=SCALE * 2)

    # Motor Hub
    hub_r = SCALE * 13
    draw.ellipse([x_m - hub_r, y_m - hub_r, x_m + hub_r, y_m + hub_r],
                 fill=(30, 41, 59, 255), outline=(148, 163, 184, 255), width=SCALE * 2)
    
    # LED Navigation indicator in center of motor
    led_r = SCALE * 5
    draw.ellipse([x_m - led_r, y_m - led_r, x_m + led_r, y_m + led_r],
                 fill=nav_col, outline=(255, 255, 255, 220), width=SCALE)

# 7. Central Fuselage / Canopy
body_w = int(DIM * 0.11)
body_h = int(DIM * 0.15)
# Futuristic angular drone fuselage
nose_y = cy - body_h
tail_y = cy + int(body_h * 0.9)
body_pts = [
    (cx, nose_y),                                # Nose tip
    (cx + int(body_w * 0.7), nose_y + int(body_h * 0.3)), # Forward-right
    (cx + body_w, cy),                           # Mid-right
    (cx + int(body_w * 0.8), cy + int(body_h * 0.6)),   # Aft-right
    (cx + int(body_w * 0.4), tail_y),            # Tail-right
    (cx - int(body_w * 0.4), tail_y),            # Tail-left
    (cx - int(body_w * 0.8), cy + int(body_h * 0.6)),   # Aft-left
    (cx - body_w, cy),                           # Mid-left
    (cx - int(body_w * 0.7), nose_y + int(body_h * 0.3)), # Forward-left
]

# Body shadow
draw.polygon([(x + SCALE*3, y + SCALE*3) for x, y in body_pts], fill=(0, 0, 0, 140))
# Body base
draw.polygon(body_pts, fill=(30, 41, 59, 255), outline=(100, 116, 139, 255))

# Canopy inner facet
canopy_pts = [
    (cx, nose_y + SCALE*8),
    (cx + int(body_w * 0.5), cy - SCALE*4),
    (cx + int(body_w * 0.25), cy + int(body_h * 0.4)),
    (cx - int(body_w * 0.25), cy + int(body_h * 0.4)),
    (cx - int(body_w * 0.5), cy - SCALE*4)
]
draw.polygon(canopy_pts, fill=(15, 23, 42, 255), outline=(0, 229, 255, 200), width=SCALE)

# Glowing core / AI reactor on drone
core_r = SCALE * 9
draw.ellipse([cx - core_r, cy - core_r, cx + core_r, cy + core_r],
             fill=(0, 240, 255, 255), outline=(255, 255, 255, 240), width=SCALE)
core_inner = SCALE * 4
draw.ellipse([cx - core_inner, cy - core_inner, cx + core_inner, cy + core_inner],
             fill=(255, 255, 255, 255))

# Front FPV Camera Lens
cam_y = nose_y + SCALE * 4
cam_r = SCALE * 5
draw.ellipse([cx - cam_r, cam_y - cam_r, cx + cam_r, cam_y + cam_r],
             fill=(0, 229, 255, 255), outline=(255, 255, 255, 220), width=SCALE)

# 8. Downscale to final 512x512 with high-quality Lanczos resampling
final_img = img.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
final_img.save("/home/piet/ADAPT-X/assets/drone_logo.png", format="PNG")
print("Successfully generated /home/piet/ADAPT-X/assets/drone_logo.png")
