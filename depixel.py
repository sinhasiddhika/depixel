import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from io import BytesIO
import cv2
from sklearn.cluster import KMeans
import scipy.ndimage as ndi
from skimage import morphology, filters, restoration, segmentation
import random
import colorsys

# Set page config
st.set_page_config(page_title="Professional Carpet Design Converter", layout="wide")

# App title
st.title("🪄 Professional Carpet Design Converter")
st.markdown("Transform *pixel art carpet designs* into *photorealistic carpet textures* with professional accuracy!")

# Image uploader
uploaded_file = st.file_uploader("Upload your carpet pixel art design", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load image using PIL
        pixel_image = Image.open(uploaded_file)
        
        # Convert to RGB if needed
        if pixel_image.mode in ('RGBA', 'LA', 'P'):
            pixel_image = pixel_image.convert('RGB')
        
        # Show original pixel art
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Carpet Design")
            st.image(pixel_image, caption=f"Size: {pixel_image.width}×{pixel_image.height}", use_container_width=True)
        
        # Get original dimensions
        orig_width, orig_height = pixel_image.size
        
        # Enhancement parameters
        st.subheader("🎛 Professional Carpet Enhancement Controls")
        
        # Carpet Material and Pattern Controls
        with st.expander("🧵 Carpet Material & Pattern Settings", expanded=True):
            col_mat1, col_mat2, col_mat3 = st.columns(3)
            
            with col_mat1:
                carpet_type = st.selectbox(
                    "Carpet Type",
                    ["Persian/Oriental Rug", "Modern Geometric", "Traditional Handwoven", "Berber Style", "Vintage Pattern", "Contemporary Loop"],
                    help="Choose the carpet style for accurate texture simulation"
                )
                
                pile_height = st.slider(
                    "Pile Height Simulation",
                    min_value=0.1,
                    max_value=3.0,
                    value=1.2,
                    step=0.1,
                    help="Simulates carpet pile height and depth"
                )
            
            with col_mat2:
                fiber_type = st.selectbox(
                    "Fiber Type",
                    ["Wool", "Cotton", "Silk", "Synthetic", "Jute", "Mixed Blend"],
                    help="Different fibers create different textures and sheens"
                )
                
                weave_density = st.slider(
                    "Weave Density",
                    min_value=50,
                    max_value=500,
                    value=200,
                    step=10,
                    help="Higher density = finer, more detailed texture"
                )
            
            with col_mat3:
                pattern_preservation = st.slider(
                    "Pattern Preservation",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.5,
                    step=0.1,
                    help="How strictly to preserve geometric patterns"
                )
                
                edge_definition = st.slider(
                    "Edge Definition",
                    min_value=0.1,
                    max_value=2.0,
                    value=1.3,
                    step=0.1,
                    help="Sharpness of pattern boundaries"
                )
        
        # Color Palette Control
        with st.expander("🎨 Advanced Color Palette Control", expanded=True):
            col_color1, col_color2 = st.columns(2)
            
            with col_color1:
                color_enhancement = st.selectbox(
                    "Color Enhancement Mode",
                    ["Preserve Original", "Enhance Saturation", "Vintage/Aged", "Vibrant Modern", "Muted Traditional"],
                    help="Different color processing approaches"
                )
                
                color_variation = st.slider(
                    "Natural Color Variation",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                    help="Adds natural color variations found in real carpets"
                )
            
            with col_color2:
                palette_extraction = st.checkbox("🔍 Auto Extract Color Palette", value=True)
                palette_size = st.slider(
                    "Color Palette Size",
                    min_value=3,
                    max_value=16,
                    value=8,
                    step=1,
                    help="Number of dominant colors to extract and enhance"
                )
                
                yarn_irregularity = st.slider(
                    "Yarn Color Irregularity",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.15,
                    step=0.05,
                    help="Simulates natural yarn color variations"
                )
        
        # Output Configuration
        with st.expander("📐 Output Configuration", expanded=True):
            col_out1, col_out2 = st.columns(2)
            
            with col_out1:
                upscale_factor = st.selectbox(
                    "Upscale Factor",
                    [4, 6, 8, 10, 12, 16, 20],
                    index=2,  # Default to 8x
                    help="Higher values = larger, more detailed output"
                )
                target_width = orig_width * upscale_factor
                target_height = orig_height * upscale_factor
                st.info(f"*Output Size:* {target_width} × {target_height} pixels")
            
            with col_out2:
                lighting_angle = st.slider(
                    "Lighting Angle (degrees)",
                    min_value=0,
                    max_value=360,
                    value=45,
                    step=15,
                    help="Direction of simulated lighting"
                )
                
                shadow_intensity = st.slider(
                    "Shadow Depth",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.1,
                    help="Intensity of pile shadows"
                )

        def extract_dominant_colors(img_array, n_colors=8):
            """Extract dominant colors using advanced clustering"""
            # Reshape image to list of pixels
            pixels = img_array.reshape(-1, 3)
            
            # Remove duplicate colors to improve clustering
            unique_pixels = np.unique(pixels, axis=0)
            
            if len(unique_pixels) <= n_colors:
                return unique_pixels
            
            # Use K-means clustering to find dominant colors
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers (dominant colors)
            dominant_colors = kmeans.cluster_centers_.astype(np.uint8)
            
            # Sort by frequency
            labels = kmeans.labels_
            color_counts = np.bincount(labels)
            sorted_indices = np.argsort(color_counts)[::-1]
            
            return dominant_colors[sorted_indices]

        def enhance_color_palette(colors, enhancement_mode, fiber_type):
            """Enhance color palette based on carpet type and fiber"""
            enhanced_colors = colors.copy().astype(np.float32)
            
            for i, color in enumerate(enhanced_colors):
                r, g, b = color / 255.0
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                
                if enhancement_mode == "Enhance Saturation":
                    s = min(1.0, s * 1.3)
                    v = min(1.0, v * 1.1)
                elif enhancement_mode == "Vintage/Aged":
                    s *= 0.7
                    v *= 0.8
                    # Add slight brown tint
                    h = (h + 0.05) % 1.0 if h > 0.1 else h
                elif enhancement_mode == "Vibrant Modern":
                    s = min(1.0, s * 1.4)
                    v = min(1.0, v * 1.2)
                elif enhancement_mode == "Muted Traditional":
                    s *= 0.8
                    v *= 0.9
                
                # Fiber-specific adjustments
                if fiber_type == "Silk":
                    s = min(1.0, s * 1.2)
                    v = min(1.0, v * 1.15)  # Silk has natural sheen
                elif fiber_type == "Wool":
                    s *= 0.9
                    v *= 0.95  # Wool is slightly muted
                elif fiber_type == "Cotton":
                    s *= 0.85
                    v *= 0.92  # Cotton is more matte
                
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                enhanced_colors[i] = [r * 255, g * 255, b * 255]
            
            return enhanced_colors.astype(np.uint8)

        def create_carpet_fiber_texture(width, height, fiber_type, weave_density, pile_height):
            """Generate realistic carpet fiber texture"""
            
            # Base texture resolution
            tex_scale = max(1, int(weave_density / 100))
            
            if fiber_type == "Wool":
                # Wool fibers are irregular and slightly twisted
                texture = np.random.normal(0, 0.1 * pile_height, (height, width))
                
                # Add wool fiber direction variations
                for _ in range(int(weave_density / 50)):
                    # Random fiber direction
                    angle = np.random.uniform(0, 2 * np.pi)
                    length = np.random.randint(3, 8)
                    
                    # Random starting point
                    start_x = np.random.randint(0, width - length)
                    start_y = np.random.randint(0, height - length)
                    
                    # Create fiber trace
                    for i in range(length):
                        x = int(start_x + i * np.cos(angle))
                        y = int(start_y + i * np.sin(angle))
                        if 0 <= x < width and 0 <= y < height:
                            texture[y, x] += 0.1 * pile_height
            
            elif fiber_type == "Silk":
                # Silk has smooth, parallel fibers with high sheen
                texture = np.zeros((height, width), dtype=np.float32)
                
                # Create parallel fiber pattern
                fiber_spacing = max(1, int(10 / tex_scale))
                for i in range(0, width, fiber_spacing):
                    # Add slight wave to silk fibers
                    wave = np.sin(np.arange(height) * 0.1) * 0.5
                    for j, w in enumerate(wave):
                        x = int(i + w)
                        if 0 <= x < width:
                            texture[j, x] += 0.2 * pile_height
                
                # Add silk sheen variation
                sheen_noise = np.random.normal(0, 0.05 * pile_height, (height, width))
                texture += sheen_noise
            
            elif fiber_type == "Cotton":
                # Cotton has a more uniform, matte texture
                texture = np.random.normal(0, 0.08 * pile_height, (height, width))
                
                # Add cotton fiber clumping
                for _ in range(int(weave_density / 30)):
                    center_x = np.random.randint(0, width)
                    center_y = np.random.randint(0, height)
                    radius = np.random.randint(2, 5)
                    
                    y_indices, x_indices = np.ogrid[:height, :width]
                    mask = (x_indices - center_x)**2 + (y_indices - center_y)**2 <= radius**2
                    texture[mask] += 0.05 * pile_height
            
            else:  # Synthetic or other
                # Synthetic fibers are more uniform
                texture = np.random.normal(0, 0.06 * pile_height, (height, width))
            
            # Apply weave pattern
            weave_pattern = np.zeros((height, width), dtype=np.float32)
            spacing = max(2, int(20 / tex_scale))
            
            # Horizontal weave
            for y in range(0, height, spacing):
                thickness = max(1, int(spacing * 0.6))
                for t in range(thickness):
                    if y + t < height:
                        weave_pattern[y + t, :] += 0.1 * pile_height
            
            # Vertical weave
            for x in range(0, width, spacing):
                thickness = max(1, int(spacing * 0.6))
                for t in range(thickness):
                    if x + t < width:
                        weave_pattern[:, x + t] += 0.1 * pile_height
            
            # Combine textures
            final_texture = texture + weave_pattern * 0.5
            
            # Normalize
            final_texture = (final_texture - final_texture.min()) / (final_texture.max() - final_texture.min() + 1e-8)
            
            return final_texture

        def preserve_geometric_patterns(img_array, upscaled_array, pattern_preservation):
            """Preserve geometric patterns during upscaling"""
            
            # Convert to grayscale for edge detection
            orig_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            upscaled_gray = cv2.cvtColor(upscaled_array, cv2.COLOR_RGB2GRAY)
            
            # Detect edges in original
            orig_edges = cv2.Canny(orig_gray, 30, 100)
            
            # Upscale edges
            scale_x = upscaled_array.shape[1] / img_array.shape[1]
            scale_y = upscaled_array.shape[0] / img_array.shape[0]
            
            upscaled_edges = cv2.resize(orig_edges, 
                                      (upscaled_array.shape[1], upscaled_array.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
            
            # Enhance pattern boundaries
            kernel = np.ones((3, 3), np.uint8)
            enhanced_edges = cv2.morphologyEx(upscaled_edges, cv2.MORPH_CLOSE, kernel)
            
            # Apply edge enhancement to preserve patterns
            edge_mask = enhanced_edges > 0
            
            # Sharpen along edges
            sharpened = upscaled_array.copy().astype(np.float32)
            for i in range(3):  # For each color channel
                channel = sharpened[:, :, i]
                # Apply stronger sharpening along edges
                laplacian = cv2.Laplacian(channel, cv2.CV_32F)
                channel[edge_mask] += laplacian[edge_mask] * pattern_preservation * 0.5
                sharpened[:, :, i] = np.clip(channel, 0, 255)
            
            return sharpened.astype(np.uint8)

        def apply_carpet_lighting(img_array, texture_map, lighting_angle, shadow_intensity, pile_height):
            """Apply realistic carpet lighting with pile shadows"""
            
            h, w = img_array.shape[:2]
            
            # Convert angle to radians
            angle_rad = np.radians(lighting_angle)
            light_dir_x = np.cos(angle_rad)
            light_dir_y = np.sin(angle_rad)
            
            # Create lighting map
            lighting = np.ones((h, w), dtype=np.float32)
            
            # Calculate surface normals from texture (simulating pile direction)
            grad_x = cv2.Sobel(texture_map, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(texture_map, cv2.CV_32F, 0, 1, ksize=3)
            
            # Calculate lighting intensity based on surface normal and light direction
            for y in range(h):
                for x in range(w):
                    # Surface normal (simplified)
                    normal_x = grad_x[y, x] * pile_height
                    normal_y = grad_y[y, x] * pile_height
                    normal_z = 1.0  # Assuming mostly upward-facing surface
                    
                    # Normalize normal vector
                    normal_mag = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
                    if normal_mag > 0:
                        normal_x /= normal_mag
                        normal_y /= normal_mag
                        normal_z /= normal_mag
                    
                    # Light direction (coming from above at angle)
                    light_z = np.sqrt(1 - light_dir_x**2 - light_dir_y**2)
                    
                    # Dot product for lighting intensity
                    dot_product = normal_x * light_dir_x + normal_y * light_dir_y + normal_z * light_z
                    
                    # Apply lighting
                    lighting[y, x] = max(0.3, min(1.0, 0.7 + dot_product * 0.3))
            
            # Apply shadows based on pile height variations
            shadow_map = np.ones((h, w), dtype=np.float32)
            
            # Create shadows in areas where pile would block light
            shadow_kernel = np.ones((3, 3), np.float32) / 9
            blurred_texture = cv2.filter2D(texture_map, -1, shadow_kernel)
            
            shadow_mask = texture_map < blurred_texture - 0.1
            shadow_map[shadow_mask] *= (1.0 - shadow_intensity * 0.5)
            
            # Apply lighting to image
            lit_image = img_array.copy().astype(np.float32)
            combined_lighting = lighting * shadow_map
            
            for i in range(3):
                lit_image[:, :, i] *= combined_lighting
            
            return np.clip(lit_image, 0, 255).astype(np.uint8)

        def add_yarn_irregularity(img_array, color_palette, irregularity_factor):
            """Add natural yarn color irregularities"""
            
            if irregularity_factor <= 0:
                return img_array
            
            h, w = img_array.shape[:2]
            result = img_array.copy().astype(np.float32)
            
            # For each pixel, slightly vary its color within the palette
            for y in range(h):
                for x in range(w):
                    original_color = result[y, x]
                    
                    # Find closest palette color
                    distances = np.sum((color_palette - original_color)**2, axis=1)
                    closest_idx = np.argmin(distances)
                    closest_color = color_palette[closest_idx]
                    
                    # Add slight variation
                    if np.random.random() < irregularity_factor:
                        variation = np.random.normal(0, 10 * irregularity_factor, 3)
                        varied_color = closest_color + variation
                        
                        # Blend with original
                        blend_factor = 0.3
                        result[y, x] = (1 - blend_factor) * original_color + blend_factor * varied_color
            
            return np.clip(result, 0, 255).astype(np.uint8)

        def create_professional_carpet_image(pixel_img, target_width, target_height, 
                                           carpet_type, fiber_type, pile_height, weave_density,
                                           pattern_preservation, edge_definition, 
                                           color_enhancement, color_variation, palette_size,
                                           yarn_irregularity, lighting_angle, shadow_intensity,
                                           palette_extraction):
            """Create professional-quality realistic carpet image"""
            
            try:
                img_array = np.array(pixel_img)
                
                # Step 1: Extract and enhance color palette
                if palette_extraction:
                    dominant_colors = extract_dominant_colors(img_array, palette_size)
                    enhanced_colors = enhance_color_palette(dominant_colors, color_enhancement, fiber_type)
                else:
                    enhanced_colors = extract_dominant_colors(img_array, palette_size)
                
                # Step 2: Advanced upscaling with pattern preservation
                # First, do careful upscaling
                upscaled = cv2.resize(img_array, (target_width, target_height), 
                                    interpolation=cv2.INTER_NEAREST)  # Preserve sharp edges initially
                
                # Apply slight smoothing only if upscale factor is very high
                if target_width / img_array.shape[1] > 8:
                    upscaled = cv2.medianBlur(upscaled, 3)
                
                # Step 3: Preserve geometric patterns
                upscaled = preserve_geometric_patterns(img_array, upscaled, pattern_preservation)
                
                # Step 4: Create realistic carpet texture
                carpet_texture = create_carpet_fiber_texture(
                    target_width, target_height, fiber_type, weave_density, pile_height
                )
                
                # Step 5: Apply texture to image
                textured_image = upscaled.copy().astype(np.float32)
                
                # Modulate image with carpet texture
                for i in range(3):
                    texture_effect = 1.0 + (carpet_texture - 0.5) * 0.4 * pile_height
                    textured_image[:, :, i] *= texture_effect
                
                # Step 6: Add yarn irregularities
                if yarn_irregularity > 0:
                    textured_image = add_yarn_irregularity(
                        textured_image.astype(np.uint8), enhanced_colors, yarn_irregularity
                    ).astype(np.float32)
                
                # Step 7: Apply realistic lighting
                lit_image = apply_carpet_lighting(
                    textured_image.astype(np.uint8), carpet_texture, 
                    lighting_angle, shadow_intensity, pile_height
                )
                
                # Step 8: Final color enhancement
                if color_variation > 0:
                    # Add subtle color variations
                    for i in range(3):
                        color_noise = np.random.normal(1.0, color_variation * 0.1, lit_image.shape[:2])
                        lit_image[:, :, i] = lit_image[:, :, i] * color_noise
                
                # Step 9: Edge enhancement for pattern definition
                if edge_definition > 1.0:
                    # Sharpen edges without affecting overall texture
                    kernel = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                     [-1,-1,-1]]) * (edge_definition - 1.0) * 0.1
                    
                    for i in range(3):
                        enhanced_channel = cv2.filter2D(lit_image[:, :, i].astype(np.float32), -1, kernel)
                        lit_image[:, :, i] = np.clip(enhanced_channel, 0, 255)
                
                # Final clipping and conversion
                result = np.clip(lit_image, 0, 255).astype(np.uint8)
                
                return Image.fromarray(result), enhanced_colors
                
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                # Fallback
                return pixel_img.resize((target_width, target_height), Image.LANCZOS), []

        # Generate the realistic carpet image
        if st.button("🚀 Generate Professional Carpet Design", type="primary"):
            with st.spinner("🔄 Converting to realistic carpet texture... This may take a moment for high-quality results."):
                try:
                    realistic_image, color_palette = create_professional_carpet_image(
                        pixel_image, target_width, target_height, carpet_type, fiber_type,
                        pile_height, weave_density, pattern_preservation, edge_definition,
                        color_enhancement, color_variation, palette_size, yarn_irregularity,
                        lighting_angle, shadow_intensity, palette_extraction
                    )
                    
                    # Store results
                    st.session_state.realistic_image = realistic_image
                    st.session_state.color_palette = color_palette
                    st.session_state.carpet_specs = {
                        'type': carpet_type,
                        'fiber': fiber_type,
                        'size': f"{target_width}×{target_height}",
                        'upscale_factor': upscale_factor
                    }
                    
                except Exception as e:
                    st.error(f"Failed to process image: {str(e)}")

        # Display results
        if hasattr(st.session_state, 'realistic_image') and st.session_state.realistic_image:
            with col2:
                st.subheader("Generated Realistic Carpet")
                st.image(
                    st.session_state.realistic_image,
                    caption=f"{st.session_state.carpet_specs['type']} - {st.session_state.carpet_specs['fiber']} Fiber - {st.session_state.carpet_specs['size']}",
                    use_container_width=True
                )
            
            # Show extracted color palette
            if hasattr(st.session_state, 'color_palette') and len(st.session_state.color_palette) > 0:
                st.subheader("🎨 Extracted Color Palette")
                palette_cols = st.columns(min(8, len(st.session_state.color_palette)))
                
                for i, color in enumerate(st.session_state.color_palette[:8]):
                    with palette_cols[i]:
                        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                        st.color_picker(f"Color {i+1}", color_hex, disabled=True, key=f"color_{i}")
                        st.caption(f"RGB({color[0]}, {color[1]}, {color[2]})")
            
            # Download section
            st.subheader("💾 Download Professional Carpet Design")
            
            col_d1, col_d2, col_d3 = st.columns(3)
            
            with col_d1:
                buf_png = BytesIO()
                st.session_state.realistic_image.save(buf_png, format="PNG")
                st.download_button(
                    label="📱 Download PNG (Lossless)",
                    data=buf_png.getvalue(),
                    file_name=f"professional_carpet_{carpet_type.lower().replace('/', '_')}.png",
                    mime="image/png"
                )
            
            with col_d2:
                buf_jpg = BytesIO()
                st.session_state.realistic_image.save(buf_jpg, format="JPEG", quality=95)
                st.download_button(
                    label="🖼 Download JPEG (High Quality)",
                    data=buf_jpg.getvalue(),
                    file_name=f"professional_carpet_{carpet_type.lower().replace('/', '_')}.jpg",
                    mime="image/jpeg"
                )
            
            with col_d3:
                # Create a specification sheet
                specs_text = f"""Professional Carpet Design Specifications

Design Type: {st.session_state.carpet_specs['type']}
Fiber Material: {st.session_state.carpet_specs['fiber']}
Output Resolution: {st.session_state.carpet_specs['size']} pixels
Upscale Factor: {st.session_state.carpet_specs['upscale_factor']}x

Color Palette:"""
                
                if hasattr(st.session_state, 'color_palette'):
                    for i, color in enumerate(st.session_state.color_palette[:8]):
                        specs_text += f"\nColor {i+1}: RGB({color[0]}, {color[1]}, {color[2]}) #{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                
                st.download_button(
                    label="📋 Download Specifications",
                    data=specs_text,
                    file_name=f"carpet_specs_{carpet_type.lower().replace('/', '_')}.txt",
                    mime="text/plain"
                )
    
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

else:
    st.info("⬆ Please upload a carpet pixel art design to start the professional conversion!")
    
    st.subheader("🏆 Professional Features:")
    st.markdown("""
    - **Geometric Pattern Preservation**: Maintains sharp, accurate pattern boundaries
    - **Advanced Fiber Simulation**: Wool, silk, cotton, and synthetic fiber textures
    - **Professional Color Palette Control**: Extract and enhance dominant colors
    - **Realistic Carpet Lighting**: Pile shadows and directional lighting
    - **Multiple Carpet Styles**: Persian, modern geometric, traditional handwoven
    - **Yarn Irregularity Simulation**: Natural color variations found in real carpets
    - **High-Resolution Output**: Up to 20x upscaling with pattern integrity
    - **Industry-Standard Export**: PNG, JPEG, and specification sheets
    """)
    
    st.subheader("💡 Usage Tips:")
    st.markdown("""
    - Upload high-contrast pixel art for best pattern recognition
    - Use 'Persian/Oriental Rug' for traditional geometric patterns
    - Increase 'Pattern Preservation' for sharp geometric designs
    - Higher 'Weave Density' creates finer, more detailed textures
    - Adjust 'Pile Height' to simulate different carpet thicknesses
    """)
