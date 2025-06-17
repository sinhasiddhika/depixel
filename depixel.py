import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from io import BytesIO
import cv2
from sklearn.cluster import KMeans
import scipy.ndimage as ndi
from skimage import morphology, filters, restoration, feature
import random
import math

# Set page config
st.set_page_config(page_title="AI Carpet Pixel Art to Realistic Converter", layout="wide")

# App title
st.title("🏠 AI Carpet Pixel Art to Realistic Image Converter")
st.markdown("Transform carpet pixel art into photorealistic carpet textures using advanced AI-inspired techniques!")

# Image uploader
uploaded_file = st.file_uploader("Upload your carpet pixel art", type=["jpg", "jpeg", "png"])

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
            st.subheader("Original Carpet Pixel Art")
            st.image(pixel_image, caption=f"Size: {pixel_image.width}×{pixel_image.height}", use_container_width=True)
        
        # Get original dimensions
        orig_width, orig_height = pixel_image.size
        
        # Enhanced carpet-specific parameters
        st.subheader("🧶 Advanced Carpet Realism Controls")
        
        # Carpet Type Selection
        with st.expander("🏺 Carpet Type & Weave Pattern", expanded=True):
            col_carpet1, col_carpet2 = st.columns(2)
            
            with col_carpet1:
                carpet_type = st.selectbox(
                    "Carpet Type",
                    ["Flat Weave", "Loop Pile", "Cut Pile", "Berber", "Shag", "Persian/Oriental"],
                    index=0,
                    help="Different carpet types have unique fiber patterns"
                )
                
                pile_height = st.slider(
                    "Pile Height",
                    min_value=0.1,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                    help="Height of carpet fibers"
                )
                
                fiber_density = st.slider(
                    "Fiber Density",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    help="How tightly packed the fibers are"
                )
            
            with col_carpet2:
                weave_pattern = st.selectbox(
                    "Weave Pattern",
                    ["Plain Weave", "Twill Weave", "Basket Weave", "Herringbone", "Custom Pattern"],
                    help="Underlying weave structure"
                )
                
                fiber_direction_variation = st.slider(
                    "Fiber Direction Variation",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.8,
                    step=0.1,
                    help="How much fibers vary in direction"
                )
                
                yarn_twist = st.slider(
                    "Yarn Twist",
                    min_value=0.0,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Twist in individual yarn fibers"
                )
        
        # Output Size Configuration
        with st.expander("📐 Output Size Configuration", expanded=True):
            upscale_factor = st.selectbox(
                "Upscale Factor",
                [4, 6, 8, 10, 12, 16, 20, 24],
                index=2,  # Default to 8x
                help="Higher values create more detailed textures"
            )
            target_width = orig_width * upscale_factor
            target_height = orig_height * upscale_factor
            st.info(f"Output Size: {target_width} × {target_height} pixels")
        
        # Advanced Carpet Realism Options
        with st.expander("🎯 Advanced Carpet Realism", expanded=True):
            col_real1, col_real2 = st.columns(2)
            
            with col_real1:
                st.markdown("🧵 Fiber Properties:")
                
                fiber_sheen = st.slider("Fiber Sheen", 0.0, 2.0, 0.7, 0.1, help="Glossiness of fibers")
                color_blending = st.slider("Color Blending", 0.0, 2.0, 1.2, 0.1, help="How colors blend between fibers")
                wear_simulation = st.slider("Wear Simulation", 0.0, 1.0, 0.0, 0.1, help="Simulate carpet wear")
                
            with col_real2:
                st.markdown("🌟 Lighting & Depth:")
                
                lighting_angle = st.slider("Lighting Angle", 0, 360, 135, 15, help="Angle of light source")
                shadow_depth = st.slider("Shadow Depth", 0.0, 2.0, 1.0, 0.1, help="Depth of shadows between fibers")
                texture_detail = st.slider("Texture Detail", 0.5, 3.0, 2.0, 0.1, help="Level of fine detail")
        
        def create_advanced_carpet_texture(pattern_array, carpet_type, pile_height, fiber_density, 
                                         weave_pattern, fiber_direction_variation, yarn_twist, 
                                         target_size, texture_detail):
            """Create highly realistic carpet texture based on pattern"""
            target_h, target_w = target_size
            
            # Create base fiber structure
            fiber_map = np.zeros((target_h, target_w, 3), dtype=np.float32)
            
            # Calculate fiber positions based on density
            fiber_spacing = max(1, int(4 / fiber_density))
            
            if carpet_type == "Flat Weave":
                # Create flat weave pattern with interlaced fibers
                for y in range(0, target_h, fiber_spacing):
                    for x in range(0, target_w, fiber_spacing):
                        # Determine if this is a warp or weft fiber
                        is_warp = (x // fiber_spacing + y // fiber_spacing) % 2 == 0
                        
                        # Create fiber bundle
                        fiber_width = max(1, fiber_spacing // 2)
                        fiber_length = fiber_spacing * 2
                        
                        if is_warp:  # Vertical fiber
                            for fy in range(max(0, y - fiber_length//2), min(target_h, y + fiber_length//2)):
                                for fx in range(max(0, x - fiber_width//2), min(target_w, x + fiber_width//2)):
                                    # Add fiber texture
                                    fiber_intensity = 0.8 + 0.2 * np.sin(fy * 0.1 * yarn_twist)
                                    fiber_map[fy, fx] += fiber_intensity * 0.5
                        else:  # Horizontal fiber
                            for fy in range(max(0, y - fiber_width//2), min(target_h, y + fiber_width//2)):
                                for fx in range(max(0, x - fiber_length//2), min(target_w, x + fiber_length//2)):
                                    fiber_intensity = 0.8 + 0.2 * np.sin(fx * 0.1 * yarn_twist)
                                    fiber_map[fy, fx] += fiber_intensity * 0.5
            
            elif carpet_type == "Loop Pile":
                # Create loop pile texture
                loop_size = max(2, int(pile_height * 3))
                
                for y in range(0, target_h, fiber_spacing):
                    for x in range(0, target_w, fiber_spacing):
                        # Create loop structure
                        loop_center_y = y + random.randint(-fiber_spacing//4, fiber_spacing//4)
                        loop_center_x = x + random.randint(-fiber_spacing//4, fiber_spacing//4)
                        
                        # Create elliptical loop
                        for ly in range(max(0, loop_center_y - loop_size), min(target_h, loop_center_y + loop_size)):
                            for lx in range(max(0, loop_center_x - loop_size), min(target_w, loop_center_x + loop_size)):
                                # Calculate distance from loop center
                                dy = ly - loop_center_y
                                dx = lx - loop_center_x
                                
                                # Create loop shape
                                if dx*dx + dy*dy <= loop_size*loop_size:
                                    # Loop height varies
                                    loop_height = pile_height * (1 - (dx*dx + dy*dy) / (loop_size*loop_size))
                                    fiber_map[ly, lx] = max(fiber_map[ly, lx], loop_height)
            
            elif carpet_type == "Cut Pile":
                # Create cut pile texture with individual fiber ends
                for y in range(0, target_h, fiber_spacing):
                    for x in range(0, target_w, fiber_spacing):
                        # Create fiber tuft
                        tuft_size = max(1, int(pile_height * 2))
                        
                        # Add direction variation
                        direction_offset_x = int(fiber_direction_variation * random.uniform(-1, 1) * tuft_size)
                        direction_offset_y = int(fiber_direction_variation * random.uniform(-1, 1) * tuft_size)
                        
                        for fy in range(max(0, y - tuft_size), min(target_h, y + tuft_size)):
                            for fx in range(max(0, x - tuft_size), min(target_w, x + tuft_size)):
                                # Calculate fiber position with direction
                                adjusted_x = fx + direction_offset_x
                                adjusted_y = fy + direction_offset_y
                                
                                if 0 <= adjusted_y < target_h and 0 <= adjusted_x < target_w:
                                    # Fiber height varies within tuft
                                    distance = np.sqrt((fx - x)*2 + (fy - y)*2)
                                    if distance <= tuft_size:
                                        fiber_height = pile_height * (1 - distance / tuft_size) * random.uniform(0.8, 1.2)
                                        fiber_map[adjusted_y, adjusted_x] = max(fiber_map[adjusted_y, adjusted_x], fiber_height)
            
            else:  # Default texture
                # Simple fiber texture
                for y in range(0, target_h, fiber_spacing):
                    for x in range(0, target_w, fiber_spacing):
                        fiber_map[y:y+fiber_spacing, x:x+fiber_spacing] = pile_height * random.uniform(0.8, 1.2)
            
            # Add fine texture detail
            if texture_detail > 1.0:
                # Add fiber surface texture
                fine_texture = np.random.normal(0, 0.05 * texture_detail, (target_h, target_w))
                fiber_map[:, :, 0] += fine_texture
                
                # Add yarn twist detail
                twist_texture = np.zeros((target_h, target_w))
                for y in range(target_h):
                    for x in range(target_w):
                        twist_val = np.sin(x * 0.1 * yarn_twist) * np.sin(y * 0.1 * yarn_twist) * 0.02 * texture_detail
                        twist_texture[y, x] = twist_val
                
                fiber_map[:, :, 1] += twist_texture
            
            return fiber_map
        
        def apply_carpet_lighting_and_shading(img_array, fiber_map, lighting_angle, shadow_depth, fiber_sheen):
            """Apply realistic lighting and shading to carpet"""
            h, w = img_array.shape[:2]
            
            # Convert lighting angle to radians
            angle_rad = np.radians(lighting_angle)
            light_dir_x = np.cos(angle_rad)
            light_dir_y = np.sin(angle_rad)
            
            # Create lighting map
            lighting = np.ones((h, w), dtype=np.float32)
            
            # Calculate normal vectors from fiber map
            if len(fiber_map.shape) == 3:
                height_map = fiber_map[:, :, 0]
            else:
                height_map = fiber_map
            
            # Calculate gradients for normal computation
            grad_x = np.gradient(height_map, axis=1)
            grad_y = np.gradient(height_map, axis=0)
            
            # Calculate lighting based on surface normals
            for y in range(h):
                for x in range(w):
                    # Surface normal
                    normal_x = -grad_x[y, x]
                    normal_y = -grad_y[y, x]
                    normal_z = 1.0
                    
                    # Normalize
                    normal_length = np.sqrt(normal_x*2 + normal_y2 + normal_z*2)
                    if normal_length > 0:
                        normal_x /= normal_length
                        normal_y /= normal_length
                        normal_z /= normal_length
                    
                    # Calculate dot product with light direction
                    dot_product = normal_x * light_dir_x + normal_y * light_dir_y + normal_z * 0.5
                    
                    # Base lighting
                    base_light = max(0.2, dot_product)
                    
                    # Add fiber sheen (specular highlight)
                    if fiber_sheen > 0:
                        specular = max(0, dot_product) ** (10 / max(0.1, fiber_sheen))
                        base_light += specular * fiber_sheen * 0.3
                    
                    lighting[y, x] = base_light
            
            # Apply shadows based on fiber depth
            shadow_map = np.ones((h, w), dtype=np.float32)
            if shadow_depth > 0:
                # Create shadow mask from fiber height differences
                shadow_mask = ndi.maximum_filter(height_map, size=3) - height_map
                shadow_mask = shadow_mask / (shadow_mask.max() + 1e-8)  # Normalize
                shadow_map = 1.0 - shadow_mask * shadow_depth * 0.5
            
            # Apply lighting and shadows to image
            lit_image = img_array.copy().astype(np.float32)
            combined_lighting = lighting * shadow_map
            
            for i in range(3):
                lit_image[:, :, i] *= combined_lighting
            
            return np.clip(lit_image, 0, 255).astype(np.uint8)
        
        def apply_color_blending_and_variation(img_array, color_blending, wear_simulation):
            """Apply realistic color blending and wear effects"""
            h, w = img_array.shape[:2]
            result = img_array.copy().astype(np.float32)
            
            # Color blending between adjacent areas
            if color_blending > 0:
                # Apply bilateral filter for color blending
                blending_strength = min(15, max(3, int(color_blending * 5)))
                result = cv2.bilateralFilter(result.astype(np.uint8), 
                                           blending_strength, 
                                           75 * color_blending, 
                                           75 * color_blending).astype(np.float32)
            
            # Simulate wear patterns
            if wear_simulation > 0:
                # Create wear pattern (more wear in center, less at edges)
                wear_pattern = np.ones((h, w), dtype=np.float32)
                
                center_x, center_y = w // 2, h // 2
                for y in range(h):
                    for x in range(w):
                        # Distance from center
                        dist_from_center = np.sqrt((x - center_x)*2 + (y - center_y)*2)
                        max_dist = np.sqrt(center_x*2 + center_y*2)
                        
                        # More wear in center
                        wear_factor = 1.0 - (dist_from_center / max_dist) * 0.3
                        
                        # Add random wear spots
                        if random.random() < 0.001 * wear_simulation:
                            wear_factor *= 0.7
                        
                        wear_pattern[y, x] = wear_factor
                
                # Apply wear to colors (make them slightly duller)
                for i in range(3):
                    result[:, :, i] *= (1.0 - wear_simulation * 0.2 * (1 - wear_pattern))
            
            return np.clip(result, 0, 255).astype(np.uint8)
        
        def create_ultra_realistic_carpet(pixel_img, target_width, target_height, carpet_type,
                                        pile_height, fiber_density, weave_pattern, 
                                        fiber_direction_variation, yarn_twist, fiber_sheen,
                                        color_blending, wear_simulation, lighting_angle,
                                        shadow_depth, texture_detail):
            """Create ultra-realistic carpet image from pixel art"""
            
            try:
                img_array = np.array(pixel_img)
                
                # Step 1: Upscale the base image using advanced interpolation
                upscaled = cv2.resize(img_array, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                
                # Step 2: Create detailed carpet fiber texture
                fiber_map = create_advanced_carpet_texture(
                    img_array, carpet_type, pile_height, fiber_density,
                    weave_pattern, fiber_direction_variation, yarn_twist,
                    (target_height, target_width), texture_detail
                )
                
                # Step 3: Apply carpet-specific lighting and shading
                lit_image = apply_carpet_lighting_and_shading(
                    upscaled, fiber_map, lighting_angle, shadow_depth, fiber_sheen
                )
                
                # Step 4: Apply color blending and wear effects
                final_image = apply_color_blending_and_variation(
                    lit_image, color_blending, wear_simulation
                )
                
                # Step 5: Final enhancement - add subtle noise for realism
                h, w = final_image.shape[:2]
                noise = np.random.normal(0, 2, (h, w, 3))
                final_image = final_image.astype(np.float32) + noise
                final_image = np.clip(final_image, 0, 255).astype(np.uint8)
                
                # Step 6: Enhance contrast and saturation slightly
                final_pil = Image.fromarray(final_image)
                
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(final_pil)
                final_pil = enhancer.enhance(1.1)
                
                # Enhance color
                enhancer = ImageEnhance.Color(final_pil)
                final_pil = enhancer.enhance(1.05)
                
                return final_pil
                
            except Exception as e:
                st.error(f"Error during carpet processing: {str(e)}")
                # Fallback to basic upscaling
                return pixel_img.resize((target_width, target_height), Image.LANCZOS)
        
        # Generate the realistic carpet
        if st.button("🚀 Generate Ultra-Realistic Carpet", type="primary"):
            with st.spinner("🧶 Weaving your carpet with AI... Creating realistic fibers and textures..."):
                try:
                    realistic_carpet = create_ultra_realistic_carpet(
                        pixel_image, target_width, target_height, carpet_type,
                        pile_height, fiber_density, weave_pattern,
                        fiber_direction_variation, yarn_twist, fiber_sheen,
                        color_blending, wear_simulation, lighting_angle,
                        shadow_depth, texture_detail
                    )
                    
                    # Store the result in session state
                    st.session_state.realistic_carpet = realistic_carpet
                    st.session_state.target_width = target_width
                    st.session_state.target_height = target_height
                    st.session_state.carpet_type = carpet_type
                    
                    st.success("✅ Realistic carpet generated successfully!")
                    
                except Exception as e:
                    st.error(f"Failed to process carpet: {str(e)}")
                    st.session_state.realistic_carpet = pixel_image.resize((target_width, target_height), Image.LANCZOS)
        
        # Show realistic carpet if it exists in session state
        if hasattr(st.session_state, 'realistic_carpet') and st.session_state.realistic_carpet:
            with col2:
                st.subheader("Generated Realistic Carpet")
                st.image(
                    st.session_state.realistic_carpet, 
                    caption=f"Realistic {st.session_state.carpet_type}: {st.session_state.target_width}×{st.session_state.target_height}",
                    use_container_width=True
                )
            
            # Quality analysis
            st.subheader("🔍 Quality Analysis")
            col_q1, col_q2, col_q3 = st.columns(3)
            
            with col_q1:
                st.metric("Resolution", f"{st.session_state.target_width}×{st.session_state.target_height}")
            with col_q2:
                st.metric("Upscale Factor", f"{upscale_factor}x")
            with col_q3:
                st.metric("Carpet Type", st.session_state.carpet_type)
            
            # Download section
            st.subheader("💾 Download Realistic Carpet")
            
            col_d1, col_d2, col_d3 = st.columns(3)
            
            with col_d1:
                # Download PNG (highest quality)
                buf_png = BytesIO()
                st.session_state.realistic_carpet.save(buf_png, format="PNG")
                st.download_button(
                    label="📱 Download PNG (Best Quality)",
                    data=buf_png.getvalue(),
                    file_name=f"realistic_carpet_{st.session_state.carpet_type.lower().replace(' ', '_')}.png",
                    mime="image/png"
                )
            
            with col_d2:
                # Download JPEG (smaller file)
                buf_jpg = BytesIO()
                st.session_state.realistic_carpet.save(buf_jpg, format="JPEG", quality=95)
                st.download_button(
                    label="🖼 Download JPEG (Smaller)",
                    data=buf_jpg.getvalue(),
                    file_name=f"realistic_carpet_{st.session_state.carpet_type.lower().replace(' ', '_')}.jpg",
                    mime="image/jpeg"
                )
            
            with col_d3:
                # Download TIFF (professional)
                buf_tiff = BytesIO()
                st.session_state.realistic_carpet.save(buf_tiff, format="TIFF")
                st.download_button(
                    label="🎨 Download TIFF (Professional)",
                    data=buf_tiff.getvalue(),
                    file_name=f"realistic_carpet_{st.session_state.carpet_type.lower().replace(' ', '_')}.tiff",
                    mime="image/tiff"
                )
    
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

else:
    st.info("⬆ Please upload a carpet pixel art image to start the AI conversion!")
    
    st.subheader("🏠 Carpet-Specific AI Features:")
    st.markdown("""
    - *Advanced Carpet Weave Simulation*: Flat weave, loop pile, cut pile, and more
    - *Realistic Fiber Physics*: Individual fiber direction, twist, and density
    - *Professional Lighting*: Adjustable angle lighting with fiber sheen
    - *Depth and Shadow Simulation*: Realistic shadows between carpet fibers  
    - *Color Blending*: Natural color transitions between pattern areas
    - *Wear Pattern Simulation*: Realistic carpet aging and wear effects
    - *Multiple Carpet Types*: Berber, Persian, Shag, and custom patterns
    - *Ultra-High Resolution*: Up to 24x upscaling for detailed textures
    """)
    
    st.subheader("💡 Tips for Best Results:")
    st.markdown("""
    - Use *higher upscale factors* (12x-20x) for more detailed fiber textures
    - *Flat Weave* works best for geometric patterns like your example
    - Adjust *pile height* based on desired carpet thickness
    - *Fiber density* affects how tightly woven the carpet appears
    - *Lighting angle* of 135° simulates natural room lighting
    - Enable *texture detail* at 2.0+ for ultra-realistic fiber appearance
    """)
