"""
Generate ultra-photorealistic architectural and environmental textures for MuJoCo City Environment.
Includes 1024x1024 glass curtain-walls, weathered brick brownstones, obsidian corporate towers,
water caustics for the plaza fountain, water spray, propeller motion blur discs, natural tree bark,
organic foliage, asphalt road with aggregate, granite plaza mosaics, and rooftop helipads.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_all_textures(output_dir="models/textures"):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(42)

    print(f"[create_all_textures] Generating photorealistic 3D city textures in {output_dir}...")

    # --------------------------------------------------------------------------
    # 1. Modern Glass Skyscraper Facade (1024x1024)
    # Deep sky reflection, interior lit office desks, crisp aluminum mullions
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (1024, 1024), color=(22, 42, 68))
    draw = ImageDraw.Draw(img)
    n_floors = 16
    n_bays = 16
    fh = 1024 // n_floors
    bw = 1024 // n_bays

    # Sky gradient undercoat
    for y in range(1024):
        t = y / 1024.0
        r = int(18 * (1 - t) + 45 * t)
        g = int(36 * (1 - t) + 85 * t)
        b = int(62 * (1 - t) + 140 * t)
        draw.line([(0, y), (1024, y)], fill=(r, g, b))

    for f in range(n_floors):
        y0 = f * fh
        # Horizontal steel spandrel / floor slab
        draw.rectangle([0, y0, 1024, y0 + 10], fill=(14, 22, 34))
        draw.line([(0, y0 + 10), (1024, y0 + 10)], fill=(85, 115, 145), width=2)
        draw.line([(0, y0 + fh - 2), (1024, y0 + fh - 2)], fill=(45, 65, 88), width=1)

        for b in range(n_bays):
            x0 = b * bw
            # Vertical aluminum mullion
            draw.line([(x0, y0), (x0, y0 + fh)], fill=(125, 155, 185), width=3)
            draw.line([(x0 + 1, y0), (x0 + 1, y0 + fh)], fill=(175, 205, 235), width=1)

            # Window pane with architectural reflections & blinds
            win = [x0 + 5, y0 + 12, x0 + bw - 4, y0 + fh - 4]
            rand_val = rng.rand()
            if rand_val > 0.82:
                # Warm illuminated executive office
                pane_col = (235, 225, 175)
            elif rand_val > 0.55:
                # Bright daytime sky reflection
                pane_col = (75, 145, 215)
            elif rand_val > 0.25:
                # Deep azure glass
                pane_col = (38, 85, 145)
            else:
                # Dark reflective tint
                pane_col = (18, 45, 78)

            draw.rectangle(win, fill=pane_col)

            # Architectural venetian blinds on select offices
            if rand_val > 0.65:
                for blind_y in range(win[1] + 6, win[3] - 8, 8):
                    draw.line([(win[0] + 2, blind_y), (win[2] - 2, blind_y)], fill=(195, 210, 225), width=2)

    img.save(os.path.join(output_dir, "facade_glass_tower.png"))

    # --------------------------------------------------------------------------
    # 2. Heritage Red Brick Brownstone Facade (1024x1024)
    # Weathered kiln-fired bricks, mortar lines, stone window headers & sills
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (1024, 1024), color=(145, 52, 38))
    draw = ImageDraw.Draw(img)

    # Photorealistic brick bonding with mortar
    brick_h = 16
    brick_w = 32
    for y in range(0, 1024, brick_h):
        draw.line([(0, y), (1024, y)], fill=(95, 88, 82), width=2)
        shift = (brick_w // 2) if ((y // brick_h) % 2 == 1) else 0
        for x in range(shift, 1024 + brick_w, brick_w):
            draw.line([(x, y), (x, y + brick_h)], fill=(95, 88, 82), width=2)
            # Subtle individual brick color variation
            var = rng.randint(-18, 22)
            br_col = (max(80, min(185, 145 + var)), max(35, min(75, 52 + var // 2)), max(25, min(55, 38 + var // 3)))
            draw.rectangle([x - brick_w + 2, y + 2, x - 2, y + brick_h - 2], fill=br_col)

    # 6 Floors of classic residential brownstone sash windows
    n_floors = 6
    n_bays = 8
    fh = 1024 // n_floors
    bw = 1024 // n_bays

    for f in range(n_floors):
        y0 = f * fh
        for b in range(n_bays):
            x0 = b * bw
            wx0 = x0 + 26
            wx1 = x0 + bw - 26
            wy0 = y0 + 36
            wy1 = y0 + fh - 24

            # Ornate carved sandstone lintel header with keystone
            draw.rectangle([wx0 - 6, wy0 - 14, wx1 + 6, wy0 - 2], fill=(225, 218, 204))
            draw.rectangle([(wx0 + wx1)//2 - 6, wy0 - 18, (wx0 + wx1)//2 + 6, wy0 - 1], fill=(238, 232, 220))
            # Deep stone sill
            draw.rectangle([wx0 - 8, wy1 + 2, wx1 + 8, wy1 + 12], fill=(205, 198, 185))

            # Dark architectural wood window frame
            draw.rectangle([wx0, wy0, wx1, wy1], fill=(28, 30, 34))

            # Multi-pane sash division with glass reflections
            mid_x = (wx0 + wx1) // 2
            mid_y = (wy0 + wy1) // 2
            p_col = (242, 225, 175) if rng.rand() > 0.50 else (55, 75, 100)

            for pane in [
                [wx0 + 3, wy0 + 3, mid_x - 2, mid_y - 2],
                [mid_x + 2, wy0 + 3, wx1 - 3, mid_y - 2],
                [wx0 + 3, mid_y + 2, mid_x - 2, wy1 - 3],
                [mid_x + 2, mid_y + 2, wx1 - 3, wy1 - 3]
            ]:
                draw.rectangle(pane, fill=p_col)
                # Glass glare highlight
                draw.line([(pane[0] + 3, pane[1] + 3), (pane[2] - 3, pane[3] - 3)], fill=(255, 255, 255, 80), width=1)

    img.save(os.path.join(output_dir, "facade_brick_loft.png"))

    # --------------------------------------------------------------------------
    # 3. Obsidian & Champagne Bronze Corporate Tower (1024x1024)
    # Sleek dark reflective glass with bold metallic architectural fins
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (1024, 1024), color=(14, 16, 20))
    draw = ImageDraw.Draw(img)
    n_floors = 14
    n_bays = 12
    fh = 1024 // n_floors
    bw = 1024 // n_bays

    # Draw champagne-gold vertical structural pylons
    for b in range(n_bays + 1):
        x0 = b * bw
        draw.rectangle([x0 - 5, 0, x0 + 5, 1024], fill=(185, 142, 75))
        draw.line([(x0 - 2, 0), (x0 - 2, 1024)], fill=(245, 215, 145), width=2)
        draw.line([(x0 + 4, 0), (x0 + 4, 1024)], fill=(110, 85, 45), width=1)

    for f in range(n_floors):
        y0 = f * fh
        draw.rectangle([0, y0 + 16, 1024, y0 + fh - 6], fill=(22, 28, 38))
        for b in range(n_bays):
            x0 = b * bw
            win = [x0 + 8, y0 + 20, x0 + bw - 8, y0 + fh - 10]
            val = rng.rand()
            c = (215, 185, 115) if val > 0.70 else (28, 48, 72)
            draw.rectangle(win, fill=c)
            # Horizontal golden louver accents
            draw.line([(win[0], win[1] + (win[3] - win[1]) // 2), (win[2], win[1] + (win[3] - win[1]) // 2)], fill=(160, 125, 65), width=1)

    img.save(os.path.join(output_dir, "facade_modern_corporate.png"))

    # --------------------------------------------------------------------------
    # 4. Contemporary Precast White Concrete & Ribbon Windows (1024x1024)
    # Architectural cast stone panels with deep continuous ribbon glazing
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (1024, 1024), color=(228, 230, 235))
    draw = ImageDraw.Draw(img)
    n_floors = 12
    fh = 1024 // n_floors

    for f in range(n_floors):
        y0 = f * fh
        # Architectural concrete expansion joint
        draw.line([(0, y0), (1024, y0)], fill=(145, 150, 158), width=3)
        draw.line([(0, y0 + 2), (1024, y0 + 2)], fill=(250, 250, 255), width=1)

        # Recessed ribbon glazing band
        draw.rectangle([0, y0 + 24, 1024, y0 + fh - 16], fill=(16, 22, 30))

        # Vertical glass panel dividers
        for x in range(0, 1024, 48):
            draw.line([(x, y0 + 24), (x, y0 + fh - 16)], fill=(65, 72, 82), width=3)
            c = (210, 228, 245) if rng.rand() > 0.45 else (38, 55, 78)
            draw.rectangle([x + 3, y0 + 28, x + 45, y0 + fh - 20], fill=c)

    img.save(os.path.join(output_dir, "facade_concrete_panels.png"))

    # --------------------------------------------------------------------------
    # 5. Storefront Retail / Street Level Facade (1024x1024)
    # Modern street retail: large glass display windows, awnings, signage, cafe
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (1024, 1024), color=(45, 48, 54))
    draw = ImageDraw.Draw(img)
    bay_w = 1024 // 4

    awning_colors = [
        [(195, 45, 45), (245, 245, 245)],    # Red / White striped awning
        [(35, 105, 175), (245, 245, 245)],   # Blue / White striped awning
        [(45, 145, 80), (245, 245, 245)],    # Emerald Green awning
        [(215, 145, 30), (245, 245, 245)],   # Golden Bistro awning
    ]
    signs = ["METRO BISTRO", "CITY TECH LAB", "URBAN APOTHECARY", "SKYLINE CAFE"]

    for b in range(4):
        x0 = b * bay_w
        # Striped fabric awning with scalloped valance
        c_main, c_alt = awning_colors[b]
        draw.rectangle([x0 + 8, 20, x0 + bay_w - 8, 140], fill=c_main)
        for ax in range(x0 + 8, x0 + bay_w - 8, 24):
            draw.line([(ax, 20), (ax, 140)], fill=c_alt, width=10)

        # Architectural shop fascia sign
        draw.rectangle([x0 + 12, 148, x0 + bay_w - 12, 220], fill=(18, 20, 24))
        draw.rectangle([x0 + 12, 148, x0 + bay_w - 12, 220], outline=(185, 155, 85), width=2)
        # Storefront large display window (warm illuminated showroom)
        draw.rectangle([x0 + 12, 235, x0 + bay_w - 12, 980], fill=(235, 220, 185))
        draw.rectangle([x0 + 12, 235, x0 + bay_w - 12, 980], outline=(15, 18, 22), width=8)

        # Center double glass entrance doors
        mid_x = x0 + bay_w // 2
        draw.line([(mid_x, 235), (mid_x, 980)], fill=(15, 18, 22), width=6)
        # Door handles
        draw.rectangle([mid_x - 12, 600, mid_x - 6, 680], fill=(215, 185, 85))
        draw.rectangle([mid_x + 6, 600, mid_x + 12, 680], fill=(215, 185, 85))

        # Showroom display merchandise silhouette
        draw.rectangle([x0 + 24, 750, mid_x - 24, 950], fill=(160, 140, 110))
        draw.rectangle([mid_x + 24, 750, x0 + bay_w - 24, 950], fill=(160, 140, 110))

    img.save(os.path.join(output_dir, "facade_storefront.png"))

    # --------------------------------------------------------------------------
    # 6. Plaza Fountain Water Surface (1024x1024)
    # Shimmering crystal-blue water with animated caustics, wave crests & ripples
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (1024, 1024), color=(25, 125, 205))
    draw = ImageDraw.Draw(img)

    # Multi-frequency procedural caustics
    cx, cy = 512, 512
    # Ripple rings radiating from the fountain spout
    for r in range(20, 500, 24):
        alpha_val = int(max(40, 160 - r * 0.25))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(145, 225, 255), width=3)
        draw.ellipse([cx - r - 6, cy - r - 6, cx + r + 6, cy + r + 6], outline=(185, 240, 255), width=2)

    # Shimmering water caustics noise
    for _ in range(800):
        rx = rng.randint(0, 1024)
        ry = rng.randint(0, 1024)
        rw = rng.randint(12, 45)
        rh = rng.randint(6, 20)
        c_caustic = (rng.randint(120, 210), rng.randint(215, 255), 255)
        draw.ellipse([rx, ry, rx + rw, ry + rh], outline=c_caustic, width=2)

    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    img.save(os.path.join(output_dir, "fountain_water.png"))

    # --------------------------------------------------------------------------
    # 7. Water Spray & Geyser Plume Texture (512x512)
    # Aerated white water foam, crystalline water droplets, velocity spray
    # --------------------------------------------------------------------------
    img = Image.new("RGBA", (512, 512), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = 256, 256
    # Radial spray plume gradient
    for r in range(240, 0, -4):
        alpha = int(np.clip((240 - r) * 1.1, 0, 230))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(215, 242, 255, alpha))

    # Turbulent water droplets
    for _ in range(600):
        dist = rng.rand() * 220
        theta = rng.rand() * 2 * np.pi
        px = int(cx + dist * np.cos(theta))
        py = int(cy + dist * np.sin(theta))
        drop_r = rng.randint(2, 7)
        draw.ellipse([px - drop_r, py - drop_r, px + drop_r, py + drop_r], fill=(255, 255, 255, rng.randint(140, 255)))

    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    img.save(os.path.join(output_dir, "water_spray.png"))

    # --------------------------------------------------------------------------
    # 8. High-Speed Spinning Propeller Motion Blur Disc (512x512)
    # Semi-transparent radial gradient, illuminated blade tip trail & hub
    # --------------------------------------------------------------------------
    img = Image.new("RGBA", (512, 512), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = 256, 256

    # Radial sweep motion blur representing thousands of RPM
    for r in range(235, 30, -3):
        # Outer rim glows brighter like real composite carbon blades
        factor = (r - 30) / 205.0
        alpha = int(140 * factor * factor)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(245, 245, 250, alpha))

    # Bright blade tip circle rim (distinct aerodynamic tip vortex)
    draw.ellipse([cx - 238, cy - 238, cx + 238, cy + 238], outline=(255, 255, 255, 220), width=3)
    draw.ellipse([cx - 234, cy - 234, cx + 234, cy + 234], outline=(255, 235, 120, 180), width=2)

    # 2 sweeping spiral blade shadow arcs
    for k in range(2):
        base_angle = k * np.pi
        for a in np.linspace(0, np.pi * 0.7, 40):
            ang = base_angle + a
            r_arc = 40 + a * 260
            px = int(cx + r_arc * np.cos(ang))
            py = int(cy + r_arc * np.sin(ang))
            draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(40, 45, 55, int(180 * (1 - a / (np.pi * 0.7)))))

    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    img.save(os.path.join(output_dir, "prop_blur.png"))

    # --------------------------------------------------------------------------
    # 9. Natural Organic Tree Foliage (512x512)
    # Multidimensional leaf clusters, sunlight highlights, chlorophyll green depth
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (512, 512), color=(26, 75, 32))
    draw = ImageDraw.Draw(img)
    # Dense leaf clusters with varied green / golden-green shades
    for _ in range(1200):
        lx = rng.randint(0, 512)
        ly = rng.randint(0, 512)
        lw = rng.randint(8, 22)
        lh = rng.randint(8, 22)
        shade = rng.rand()
        if shade > 0.75:
            col = (68, 165, 52)   # Sunlit leaf highlight
        elif shade > 0.40:
            col = (42, 118, 45)   # Mid foliage
        elif shade > 0.15:
            col = (25, 78, 30)    # Deep shadow foliage
        else:
            col = (85, 145, 42)   # Golden-green leaf
        draw.ellipse([lx, ly, lx + lw, ly + lh], fill=col)

    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img.save(os.path.join(output_dir, "tree_leaf.png"))

    # --------------------------------------------------------------------------
    # 10. Natural Weathered Tree Bark (512x512)
    # Deep vertical furrowed bark ridges, natural oak / pine grain
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (512, 512), color=(78, 48, 28))
    draw = ImageDraw.Draw(img)
    for x in range(0, 512, 8):
        var = rng.randint(-15, 15)
        r = max(40, min(120, 78 + var))
        g = max(25, min(75, 48 + var // 2))
        b = max(15, min(45, 28 + var // 3))
        draw.line([(x, 0), (x, 512)], fill=(r, g, b), width=rng.randint(3, 7))
        # Deep bark crevice shadow
        draw.line([(x + 1, 0), (x + 1, 512)], fill=(32, 18, 10), width=1)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    img.save(os.path.join(output_dir, "tree_bark.png"))

    # --------------------------------------------------------------------------
    # 11. Asphalt Road with Fine Aggregate & Tire Markings (1024x1024)
    # Dark gray bitumen, fine gravel aggregate, road lane markings
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (1024, 1024), color=(35, 38, 42))
    draw = ImageDraw.Draw(img)
    # Asphalt noise texture
    noise = rng.randint(-12, 12, (1024, 1024, 3), dtype=np.int16)
    arr = np.clip(np.array(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    # Subtle road tire polish bands
    draw.rectangle([180, 0, 320, 1024], fill=(28, 30, 34))
    draw.rectangle([704, 0, 844, 1024], fill=(28, 30, 34))
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img.save(os.path.join(output_dir, "asphalt_road.png"))

    # --------------------------------------------------------------------------
    # 12. Grand Central Plaza Star Mosaic (1024x1024)
    # Ornate polished granite and marble plaza mosaic with 8-point compass rose
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (1024, 1024), color=(205, 208, 214))
    draw = ImageDraw.Draw(img)
    cx, cy = 512, 512

    # Polished granite tile seams
    tile_s = 64
    for x in range(0, 1024, tile_s):
        draw.line([(x, 0), (x, 1024)], fill=(155, 160, 168), width=2)
    for y in range(0, 1024, tile_s):
        draw.line([(0, y), (1024, y)], fill=(155, 160, 168), width=2)

    # Concentric circular marble borders
    for r in [480, 380, 280, 180, 80]:
        c = (140, 148, 162) if r % 160 == 0 else (225, 228, 235)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c, outline=(85, 92, 105), width=5)

    # 8-Point Compass Rose Star
    pts_outer = 340
    pts_inner = 120
    for k in range(8):
        angle = k * (np.pi / 4)
        angle_next = angle + (np.pi / 8)
        p1 = (cx, cy)
        p2 = (cx + pts_outer * np.cos(angle), cy + pts_outer * np.sin(angle))
        p3 = (cx + pts_inner * np.cos(angle_next), cy + pts_inner * np.sin(angle_next))
        c_star = (38, 52, 78) if k % 2 == 0 else (245, 195, 75)
        draw.polygon([p1, p2, p3], fill=c_star)
        draw.line([p1, p2], fill=(255, 255, 255), width=2)

    img.save(os.path.join(output_dir, "plaza_mosaic.png"))

    # --------------------------------------------------------------------------
    # 13. Sidewalk Paver Tiles (1024x1024)
    # Natural granite pavers with beveled edges and joint sand
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (1024, 1024), color=(185, 188, 195))
    draw = ImageDraw.Draw(img)
    tile_size = 64
    for x in range(0, 1024, tile_size):
        draw.line([(x, 0), (x, 1024)], fill=(115, 120, 128), width=3)
    for y in range(0, 1024, tile_size):
        draw.line([(0, y), (1024, y)], fill=(115, 120, 128), width=3)

    for x in range(0, 1024, tile_size):
        for y in range(0, 1024, tile_size):
            if rng.rand() > 0.65:
                draw.rectangle([x + 3, y + 3, x + tile_size - 3, y + tile_size - 3], fill=(172, 175, 182))
            elif rng.rand() > 0.40:
                draw.rectangle([x + 3, y + 3, x + tile_size - 3, y + tile_size - 3], fill=(196, 200, 208))

    img.save(os.path.join(output_dir, "sidewalk_pavers.png"))

    # --------------------------------------------------------------------------
    # 14. Official Skyscraper Rooftop Helipad (1024x1024)
    # Heavy-duty non-slip tarmac, yellow/black hazard perimeter, white "H"
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (1024, 1024), color=(36, 40, 46))
    draw = ImageDraw.Draw(img)
    # Yellow hazard perimeter border
    draw.rectangle([16, 16, 1008, 1008], outline=(245, 195, 30), width=16)
    # Red landing circle ring
    draw.ellipse([90, 90, 934, 934], fill=(48, 52, 60), outline=(215, 45, 45), width=24)
    # Inner yellow circle
    draw.ellipse([150, 150, 874, 874], outline=(245, 205, 35), width=14)
    # Center Bold White "H"
    # Left bar
    draw.rectangle([380, 320, 440, 704], fill=(255, 255, 255))
    # Right bar
    draw.rectangle([584, 320, 644, 704], fill=(255, 255, 255))
    # Cross bar
    draw.rectangle([430, 470, 594, 554], fill=(255, 255, 255))

    img.save(os.path.join(output_dir, "helipad.png"))

    # --------------------------------------------------------------------------
    # 15. Rooftop Gravel & Mechanical Tar Surface (1024x1024)
    # --------------------------------------------------------------------------
    img = Image.new("RGB", (1024, 1024), color=(52, 56, 62))
    draw = ImageDraw.Draw(img)
    noise = rng.randint(-18, 18, (1024, 1024, 3), dtype=np.int16)
    arr = np.clip(np.array(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    for x in range(0, 1024, 256):
        draw.line([(x, 0), (x, 1024)], fill=(25, 28, 32), width=4)
    for y in range(0, 1024, 256):
        draw.line([(0, y), (1024, y)], fill=(25, 28, 32), width=4)
    draw.rectangle([420, 420, 604, 604], fill=(88, 94, 102), outline=(24, 26, 30), width=6)
    img.save(os.path.join(output_dir, "roof_tar.png"))

    print(f"[create_all_textures] Successfully generated 15 ultra-photorealistic textures in {output_dir}!")


if __name__ == "__main__":
    create_all_textures()
