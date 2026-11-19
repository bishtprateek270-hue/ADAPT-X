"""
Generate crisp, photorealistic architectural textures for MuJoCo City Environment.
Includes glass curtain-walls, brick facades, modern corporate towers, storefronts,
rooftop helipads, plaza mosaics, and sidewalk pavers.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_all_textures(output_dir="models/textures"):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(101)

    # --------------------------------------------------------------------------
    # 1. Modern Glass Skyscraper Facade (512x512)
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (512, 512), color=(24, 44, 70))
    draw = ImageDraw.Draw(img)
    n_floors = 16
    n_bays = 16
    floor_h = 512 // n_floors
    bay_w = 512 // n_bays

    for f in range(n_floors):
        y0 = f * floor_h
        # Horizontal steel floor slab / spandrel
        draw.rectangle([0, y0, 512, y0 + 6], fill=(16, 26, 42))
        draw.line([(0, y0 + 6), (512, y0 + 6)], fill=(75, 95, 120), width=1)
        for b in range(n_bays):
            x0 = b * bay_w
            # Vertical aluminum mullion
            draw.line([(x0, y0), (x0, y0 + floor_h)], fill=(110, 135, 160), width=2)
            # Window pane
            win = [x0 + 3, y0 + 7, x0 + bay_w - 2, y0 + floor_h - 2]
            v = rng.rand()
            if v > 0.85:
                pane_color = (220, 225, 185)  # Lit office (evening)
            elif v > 0.50:
                pane_color = (65, 130, 190)   # Sky reflection
            elif v > 0.20:
                pane_color = (40, 85, 140)    # Deep azure reflection
            else:
                pane_color = (20, 48, 80)     # Dark tint
            draw.rectangle(win, fill=pane_color)
            if v > 0.65:
                # Venetian blinds
                draw.line([(win[0], win[1] + 5), (win[2], win[1] + 5)], fill=(180, 195, 210), width=1)
                draw.line([(win[0], win[1] + 9), (win[2], win[1] + 9)], fill=(180, 195, 210), width=1)
    img.save(os.path.join(output_dir, "facade_glass_tower.png"))

    # --------------------------------------------------------------------------
    # 2. Heritage Red Brick Brownstone Facade (512x512)
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (512, 512), color=(145, 58, 42))
    draw = ImageDraw.Draw(img)
    # Brick row texture pattern
    for y in range(0, 512, 8):
        draw.line([(0, y), (512, y)], fill=(110, 42, 30), width=1)
        offset = 8 if (y // 8) % 2 == 1 else 0
        for x in range(offset, 512, 16):
            draw.line([(x, y), (x, y + 8)], fill=(110, 42, 30), width=1)

    # 4 Floors of detailed sash windows with stone sills & lintels
    n_floors = 6
    n_bays = 8
    floor_h = 512 // n_floors
    bay_w = 512 // n_bays

    for f in range(n_floors):
        y0 = f * floor_h
        for b in range(n_bays):
            x0 = b * bay_w
            wx0 = x0 + 14
            wx1 = x0 + bay_w - 14
            wy0 = y0 + 18
            wy1 = y0 + floor_h - 14

            # White stone lintel header
            draw.rectangle([wx0 - 2, wy0 - 6, wx1 + 2, wy0 - 1], fill=(225, 220, 210))
            # White stone sill
            draw.rectangle([wx0 - 3, wy1 + 1, wx1 + 3, wy1 + 6], fill=(210, 205, 195))
            # Window frame (dark charcoal)
            draw.rectangle([wx0, wy0, wx1, wy1], fill=(30, 32, 36))
            # Panes (warm interior / reflection)
            mid_x = (wx0 + wx1) // 2
            mid_y = (wy0 + wy1) // 2
            p_color = (235, 215, 160) if rng.rand() > 0.55 else (60, 75, 95)
            # 4 pane division
            draw.rectangle([wx0 + 2, wy0 + 2, mid_x - 1, mid_y - 1], fill=p_color)
            draw.rectangle([mid_x + 1, wy0 + 2, wx1 - 2, mid_y - 1], fill=p_color)
            draw.rectangle([wx0 + 2, mid_y + 1, mid_x - 1, wy1 - 2], fill=p_color)
            draw.rectangle([mid_x + 1, mid_y + 1, wx1 - 2, wy1 - 2], fill=p_color)
    img.save(os.path.join(output_dir, "facade_brick_loft.png"))

    # --------------------------------------------------------------------------
    # 3. Obsidian & Champagne Bronze Corporate Tower (512x512)
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (512, 512), color=(18, 20, 24))
    draw = ImageDraw.Draw(img)
    n_floors = 12
    n_bays = 10
    floor_h = 512 // n_floors
    bay_w = 512 // n_bays

    for b in range(n_bays):
        x0 = b * bay_w
        # Prominent Champagne Bronze vertical fin
        draw.rectangle([x0, 0, x0 + 6, 512], fill=(195, 150, 85))
        draw.line([(x0 + 1, 0), (x0 + 1, 512)], fill=(240, 200, 140), width=1)

    for f in range(n_floors):
        y0 = f * floor_h
        # Horizontal ribbon window band
        draw.rectangle([0, y0 + 10, 512, y0 + floor_h - 4], fill=(28, 35, 45))
        for b in range(n_bays):
            x0 = b * bay_w
            win = [x0 + 7, y0 + 12, x0 + bay_w - 2, y0 + floor_h - 6]
            c = (200, 175, 110) if rng.rand() > 0.75 else (35, 52, 75)
            draw.rectangle(win, fill=c)
    img.save(os.path.join(output_dir, "facade_modern_corporate.png"))

    # --------------------------------------------------------------------------
    # 4. Contemporary Precast White Concrete & Ribbon Windows (512x512)
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (512, 512), color=(222, 224, 228))
    draw = ImageDraw.Draw(img)
    n_floors = 10
    floor_h = 512 // n_floors

    for f in range(n_floors):
        y0 = f * floor_h
        # Concrete panel seam
        draw.line([(0, y0), (512, y0)], fill=(160, 164, 170), width=2)
        # Recessed continuous window ribbon
        draw.rectangle([0, y0 + 14, 512, y0 + floor_h - 10], fill=(22, 28, 36))
        # Window dividers
        for x in range(0, 512, 32):
            draw.line([(x, y0 + 14), (x, y0 + floor_h - 10)], fill=(70, 78, 88), width=2)
            c = (215, 225, 235) if rng.rand() > 0.4 else (45, 60, 80)
            draw.rectangle([x + 2, y0 + 16, x + 30, y0 + floor_h - 12], fill=c)
    img.save(os.path.join(output_dir, "facade_concrete_panels.png"))

    # --------------------------------------------------------------------------
    # 5. Storefront Retail / Street Level Facade (512x512)
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (512, 512), color=(50, 52, 58))
    draw = ImageDraw.Draw(img)
    # Granite base
    draw.rectangle([0, 0, 512, 512], fill=(42, 44, 48))
    # 4 Retail Store bays
    bay_w = 512 // 4
    for b in range(4):
        x0 = b * bay_w
        # Overhead colorful awning canopy
        awning_color = [(180, 40, 40), (35, 110, 160), (45, 140, 80), (190, 130, 30)][b]
        draw.rectangle([x0 + 4, 10, x0 + bay_w - 4, 60], fill=awning_color)
        for ax in range(x0 + 4, x0 + bay_w - 4, 12):
            draw.line([(ax, 10), (ax, 60)], fill=(240, 240, 240), width=4)
        # Shop fascia sign
        draw.rectangle([x0 + 8, 64, x0 + bay_w - 8, 110], fill=(20, 22, 26))
        # Storefront large display window (warm illuminated shop interior)
        draw.rectangle([x0 + 8, 120, x0 + bay_w - 8, 490], fill=(220, 205, 170))
        # Window frame
        draw.rectangle([x0 + 8, 120, x0 + bay_w - 8, 490], outline=(15, 17, 20), width=6)
        # Display window divider
        draw.line([(x0 + bay_w // 2, 120), (x0 + bay_w // 2, 490)], fill=(15, 17, 20), width=4)
        # Display shelf sill
        draw.rectangle([x0 + 14, 380, x0 + bay_w - 14, 480], fill=(160, 140, 110))
    img.save(os.path.join(output_dir, "facade_storefront.png"))

    # --------------------------------------------------------------------------
    # 6. Official Skyscraper Rooftop Helipad (512x512)
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (512, 512), color=(32, 35, 40))
    draw = ImageDraw.Draw(img)
    # Concrete square base with border
    draw.rectangle([8, 8, 504, 504], outline=(230, 180, 30), width=8)
    # Hazard diagonal yellow/black corner stripes
    for offset in range(-80, 80, 16):
        draw.line([(10, 60 + offset), (60 + offset, 10)], fill=(230, 180, 30), width=8)
        draw.line([(502, 60 + offset), (502 - (60 + offset), 10)], fill=(230, 180, 30), width=8)
        draw.line([(10, 452 - offset), (60 + offset, 502)], fill=(230, 180, 30), width=8)
        draw.line([(502, 452 - offset), (502 - (60 + offset), 502)], fill=(230, 180, 30), width=8)

    # Red landing circle rim
    draw.ellipse([50, 50, 462, 462], fill=(45, 48, 54), outline=(210, 40, 40), width=14)
    # Inner yellow circle
    draw.ellipse([80, 80, 432, 432], outline=(245, 205, 35), width=8)
    # Center Bold White "H"
    # Left bar
    draw.rectangle([200, 180, 230, 332], fill=(250, 250, 250))
    # Right bar
    draw.rectangle([282, 180, 312, 332], fill=(250, 250, 250))
    # Cross bar
    draw.rectangle([226, 240, 286, 272], fill=(250, 250, 250))
    # Emergency Weight Rating Label "12"
    draw.rectangle([240, 110, 272, 130], fill=(245, 205, 35))
    img.save(os.path.join(output_dir, "helipad.png"))

    # --------------------------------------------------------------------------
    # 7. Rooftop Gravel & Mechanical Tar Surface (512x512)
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (512, 512), color=(48, 52, 58))
    draw = ImageDraw.Draw(img)
    # Gravel noise
    noise = rng.randint(-18, 18, (512, 512, 3), dtype=np.int16)
    arr = np.clip(np.array(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    # Roofing expansion seams
    for x in range(0, 512, 128):
        draw.line([(x, 0), (x, 512)], fill=(28, 30, 34), width=3)
    for y in range(0, 512, 128):
        draw.line([(0, y), (512, y)], fill=(28, 30, 34), width=3)
    # Mechanical hatch
    draw.rectangle([200, 200, 312, 312], fill=(85, 90, 98), outline=(25, 28, 32), width=4)
    img.save(os.path.join(output_dir, "roof_tar.png"))

    # --------------------------------------------------------------------------
    # 8. Grand Central Plaza Star Mosaic (512x512)
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (512, 512), color=(190, 195, 200))
    draw = ImageDraw.Draw(img)
    cx, cy = 256, 256
    # Concentric rings
    for r in [240, 190, 140, 90, 40]:
        c = (140, 145, 155) if r % 80 == 0 else (215, 218, 222)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c, outline=(90, 95, 105), width=3)
    # 8-Point Compass Rose Star
    pts_outer = 170
    pts_inner = 60
    for k in range(8):
        angle = k * (np.pi / 4)
        angle_next = angle + (np.pi / 8)
        p1 = (cx, cy)
        p2 = (cx + pts_outer * np.cos(angle), cy + pts_outer * np.sin(angle))
        p3 = (cx + pts_inner * np.cos(angle_next), cy + pts_inner * np.sin(angle_next))
        c_star = (45, 55, 75) if k % 2 == 0 else (240, 205, 120)
        draw.polygon([p1, p2, p3], fill=c_star)
    img.save(os.path.join(output_dir, "plaza_mosaic.png"))

    # --------------------------------------------------------------------------
    # 9. Sidewalk Paver Tiles (512x512)
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (512, 512), color=(175, 178, 184))
    draw = ImageDraw.Draw(img)
    tile_size = 32
    for x in range(0, 512, tile_size):
        draw.line([(x, 0), (x, 512)], fill=(120, 124, 130), width=2)
    for y in range(0, 512, tile_size):
        draw.line([(0, y), (512, y)], fill=(120, 124, 130), width=2)
    # Subtle tile shade variations
    for x in range(0, 512, tile_size):
        for y in range(0, 512, tile_size):
            if rng.rand() > 0.7:
                draw.rectangle([x + 2, y + 2, x + tile_size - 2, y + tile_size - 2], fill=(162, 165, 172))
            elif rng.rand() > 0.5:
                draw.rectangle([x + 2, y + 2, x + tile_size - 2, y + tile_size - 2], fill=(188, 192, 198))
    img.save(os.path.join(output_dir, "sidewalk_pavers.png"))

    print(f"[create_all_textures] Successfully generated 9 high-res city textures in {output_dir}")

if __name__ == "__main__":
    create_all_textures()
