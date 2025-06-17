import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
from io import BytesIO
import cv2

def extract_palette(image, n_colors=3):
    arr = np.array(image.convert('RGB')).reshape(-1, 3)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_colors, n_init=8, random_state=42).fit(arr)
    palette = np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)
    return palette

def show_palette(palette):
    cols = st.columns(len(palette))
    selected = []
    for i, color in enumerate(palette):
        hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
        picked = cols[i].color_picker(f"Color {i+1}", hex_color, key=f"pal_{i}")
        rgb = tuple(int(picked.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
        selected.append(np.array(rgb, dtype=np.uint8))
    return np.array(selected)

def tufted_carpet_effect(
    src_img, upscale_factor=8, tuft_radius=0.45, yarn_noise=0.15, 
    highlight=0.85, shadow=0.35, light_angle_deg=135, palette=None,
    fiber_detail=0.15, tuft_shape="circle", pile_variation=0.09, grid_depth=0.08
):
    """
    Each pixel in src_img is replaced with a tuft (a raised yarn sphere) of its palette color.
    """
    arr = np.array(src_img.convert('RGB'))
    h, w = arr.shape[:2]
    th, tw = h*upscale_factor, w*upscale_factor
    carpet = np.zeros((th, tw, 3), dtype=np.float32)
    # Lighting direction
    angle = np.deg2rad(light_angle_deg)
    light = np.array([np.cos(angle), np.sin(angle), 1.0])
    light = light / np.linalg.norm(light)
    rr = int(np.ceil(upscale_factor*tuft_radius))
    y_grid, x_grid = np.ogrid[-rr:rr+1, -rr:rr+1]
    dist = np.sqrt(x_grid**2 + y_grid**2) / rr
    if tuft_shape == "square":
        mask = (np.abs(x_grid) <= rr) & (np.abs(y_grid) <= rr)
        z_map = (1 - np.maximum(np.abs(x_grid), np.abs(y_grid))/rr) * mask
    else:
        mask = dist <= 1
        z_map = np.sqrt(1 - np.clip(dist,0,1)**2) * mask
    # 3D normal map per tuft
    nx = x_grid / rr * mask
    ny = y_grid / rr * mask
    nz = z_map
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-6
    nx, ny, nz = nx/norm, ny/norm, nz/norm
    dot = nx*light[0] + ny*light[1] + nz*light[2]
    local_shading = shadow + (highlight-shadow)*(dot- dot.min())/(dot.max()-dot.min() + 1e-9)
    local_shading = local_shading * mask
    # Tuft core random texture
    for y in range(h):
        for x in range(w):
            color = arr[y, x]
            if palette is not None:
                distances = np.sqrt(((palette - color)**2).sum(axis=1))
                color = palette[np.argmin(distances)]
            # Yarn color variation per tuft
            jitter = np.random.normal(0, yarn_noise*255, size=(local_shading.shape[0], local_shading.shape[1], 3))
            # Fiber detail: light/dark streaks, using Perlin-like noise
            detail_map = fiber_detail * (
                np.sin((x*17 + x_grid)*0.15 + (y*7 + y_grid)*0.15) +
                np.cos((x*11 + x_grid)*0.23 - (y*19 + y_grid)*0.17)
            )
            # Pile variation
            pile = (1 + np.random.uniform(-pile_variation, pile_variation))
            tuft_rgb = np.clip(color[None,None,:] * (local_shading[:,:,None] * pile + detail_map[:,:,None]) + jitter, 0, 255)
            cy, cx = y*upscale_factor+upscale_factor//2, x*upscale_factor+upscale_factor//2
            sy, sx = local_shading.shape
            y0, x0 = cy-sy//2, cx-sx//2
            y1, x1 = y0+sy, x0+sx
            ys0, ys1 = max(0, y0), min(th, y1)
            xs0, xs1 = max(0, x0), min(tw, x1)
            t0y, t0x = ys0-y0, xs0-x0
            t1y, t1x = sy-(y1-ys1), sx-(x1-xs1)
            mask_crop = mask[t0y:t1y, t0x:t1x][...,None]
            carpet[ys0:ys1, xs0:xs1] = (
                carpet[ys0:ys1, xs0:xs1]*(1-mask_crop) + tuft_rgb[t0y:t1y, t0x:t1x]*mask_crop
            )
            # Simulate grid/weave: darken between tufts
            if grid_depth > 0:
                grid_y = int((cy)/upscale_factor)
                grid_x = int((cx)/upscale_factor)
                if (y < h-1) and (x < w-1):
                    edge_thick = int(upscale_factor*0.09)
                    # bottom edge
                    ys = cy+rr-edge_thick//2
                    ye = min(th, cy+rr+edge_thick//2)
                    xs = max(0, cx-rr)
                    xe = min(tw, cx+rr)
                    carpet[ys:ye, xs:xe, :] *= (1-grid_depth)
                    # right edge
                    ys2 = max(0, cy-rr)
                    ye2 = min(th, cy+rr)
                    xs2 = cx+rr-edge_thick//2
                    xe2 = min(tw, cx+rr+edge_thick//2)
                    carpet[ys2:ye2, xs2:xe2, :] *= (1-grid_depth)
    # Global fiber noise
    noise = np.random.normal(0, 6, size=carpet.shape)
    carpet += noise
    # Simulate lens blur for realism
    if upscale_factor > 13:
        ksize = int(upscale_factor/4)*2+1
        carpet = cv2.GaussianBlur(carpet, (ksize, ksize), sigmaX=upscale_factor/7)
    carpet = np.clip(carpet, 0, 255).astype(np.uint8)
    return Image.fromarray(carpet)

# --- Streamlit UI ---
st.set_page_config(page_title="AI Carpet Pixel Art to Realistic Converter", layout="wide")
st.title("🏠 AI Carpet Pixel Art to Realistic Image Converter")
st.markdown(
    "Transform carpet pixel art into **photorealistic carpet textures**: tufted, woven, and with real yarn detail!"
)

uploaded_file = st.file_uploader("Upload your carpet pixel art", type=["jpg", "jpeg", "png"])
if uploaded_file:
    src_img = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Carpet Pixel Art")
        st.image(src_img, caption=f"Size: {src_img.width}×{src_img.height}", use_container_width=True)
    # --- Palette ---
    st.subheader("🎨 Color Palette (edit to match your yarn colors)")
    palette = extract_palette(src_img, n_colors=3)
    user_palette = show_palette(palette)
    # --- Realism Controls ---
    st.subheader("🧶 Realistic Yarn & Carpet Controls")
    with st.expander("Carpet Realism Controls", expanded=True):
        upscale_factor = st.slider("Upscale Factor", 6, 24, 12, 1)
        tuft_radius = st.slider("Tuft Size (as % of cell)", 0.2, 0.7, 0.45, 0.01)
        yarn_noise = st.slider("Yarn Color Noise", 0.0, 0.4, 0.13, 0.01)
        fiber_detail = st.slider("Yarn Fiber Detail", 0.0, 0.6, 0.14, 0.01)
        pile_variation = st.slider("Pile Height Variation", 0.0, 0.2, 0.09, 0.01)
        grid_depth = st.slider("Grid/Weave Shadow", 0.0, 0.2, 0.08, 0.01)
        tuft_shape = st.selectbox("Tuft Shape", ["circle", "square"], index=0)
        highlight = st.slider("Highlight Intensity", 0.5, 1.0, 0.89, 0.01)
        shadow = st.slider("Shadow Intensity", 0.1, 0.7, 0.33, 0.01)
        light_angle = st.slider("Lighting Angle", 0, 360, 135, 5)
    if st.button("🚀 Generate Photorealistic Carpet"):
        with st.spinner("Rendering photorealistic yarn tufts..."):
            realistic = tufted_carpet_effect(
                src_img, upscale_factor=upscale_factor, tuft_radius=tuft_radius,
                yarn_noise=yarn_noise, highlight=highlight, shadow=shadow,
                light_angle_deg=light_angle, palette=user_palette,
                fiber_detail=fiber_detail, tuft_shape=tuft_shape,
                pile_variation=pile_variation, grid_depth=grid_depth
            )
            st.session_state.realistic_carpet = realistic
            st.session_state.result_shape = realistic.size
            st.success("Done!")
    if 'realistic_carpet' in st.session_state:
        with col2:
            st.subheader("Generated Realistic Carpet")
            st.image(st.session_state.realistic_carpet,
                     caption=f"Realistic Tufted: {st.session_state.result_shape[0]}×{st.session_state.result_shape[1]}",
                     use_container_width=True)
        st.subheader("💾 Download")
        buf = BytesIO()
        st.session_state.realistic_carpet.save(buf, format='PNG')
        st.download_button("Download PNG", buf.getvalue(), "realistic_tufted_carpet.png", "image/png")
else:
    st.info("⬆ Please upload a carpet pixel art image to start!")

st.markdown("""
---
**How this works:**  
- Each pixel is replaced by a shaded, textured, 3D yarn tuft matching your palette.
- Shadows, lighting, and fiber noise simulate real carpet structure.
- Tweak parameters for different carpet or rug looks.
- Good for geometric, cross-stitch, and tapestry styles.
---
**Tips:**  
- Use higher *Upscale Factor* for more detail  
- Edit the *Palette* to match your yarn or rug colors  
- Adjust *Tuft Size* and *Grid Shadow* for flat weave vs. plush pile effects  
- Try "circle" shape for loops, "square" for cross-stitch or needlepoint  
- Use *Lighting Angle* to set the direction of the main highlight  
""")
