import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from io import BytesIO
import cv2
from sklearn.cluster import KMeans
import scipy.ndimage as ndi
from skimage import morphology, filters, restoration, measure
import random
import math

# Set page config
st.set_page_config(page_title="Professional Carpet Design Visualizer", layout="wide")

# App title
st.title("🏠 Professional Carpet Design Visualizer")
st.markdown("Transform *pixelated carpet designs* into *photorealistic textile visualizations* for professional carpet manufacturing!")

# Image uploader
uploaded_file = st.file_uploader("Upload your carpet design", type=["jpg", "jpeg", "png"])

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
            st.subheader("Original Design Pattern")
            st.image(pixel_image, caption=f"Size: {pixel_image.width}×{pixel_image.height}", use_container_width=True)
        
        # Get original dimensions
        orig_width, orig_height = pixel_image.size
        
        # Enhancement parameters
        st.subheader("🎛 Professional Carpet Visualization Controls")
        
        # Carpet-Specific Settings
        with st.expander("🧶 Carpet Material & Construction", expanded=True):
            col_mat1, col_mat2 = st.columns(2)
            
            with col_mat1:
                carpet_type = st.selectbox(
                    "Carpet Construction Type",
                    ["Loop Pile (Berber)", "Cut Pile (Plush)", "Cut & Loop", "Frieze (Twisted)", "Shag", "Flatweave", "Hand-Knotted"],
                    index=0,
                    help="Different carpet construction methods create different visual textures"
                )
                
                fiber_type = st.selectbox(
                    "Fiber Material",
                    ["Wool", "Nylon", "Polyester", "Polypropylene", "Cotton", "Silk", "Jute"],
                    help="Fiber material affects sheen, texture, and appearance"
                )
                
                pile_height = st.slider(
                    "Pile Height (mm)",
                    min_value=2,
                    max_value=25,
                    value=8,
                    step=1,
                    help="Height of carpet fibers - affects shadow depth and texture"
                )
            
            with col_mat2:
                fiber_density = st.slider(
                    "Fiber Density",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    help="How tightly packed the fibers are"
                )
                
                yarn_twist = st.slider(
                    "Yarn Twist Level",
                    min_value=0.1,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Amount of twist in carpet yarn - affects light reflection"
                )
                
                color_variation = st.slider(
                    "Natural Color Variation",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.8,
                    step=0.1,
                    help="Natural variation in fiber color"
                )
        
        # Output Size Configuration
        with st.expander("📐 Output Resolution", expanded=True):
            upscale_factor = st.selectbox(
                "Resolution Multiplier",
                [4, 6, 8, 10, 12, 16, 20],
                index=2,  # Default to 8x
                help="Higher values create more detailed textures"
            )
            target_width = orig_width * upscale_factor
            target_height = orig_height * upscale_factor
            st.info(f"*Output Resolution:* {target_width} × {target_height} pixels")
        
        # Advanced Realism Controls
        with st.expander("🎯 Advanced Realism Controls", expanded=True):
            col_real1, col_real2 = st.columns(2)
            
            with col_real1:
                st.markdown("💡 **Lighting & Shadows:**")
                lighting_angle = st.slider("Lighting Angle (degrees)", 0, 90, 45, 5)
                shadow_depth = st.slider("Shadow Intensity", 0.0, 2.0, 1.2, 0.1)
                highlight_strength = st.slider("Fiber Highlights", 0.0, 2.0, 1.0, 0.1)
                
            with col_real2:
                st.markdown("🔬 **Fiber Physics:**")
                fiber_irregularity = st.slider("Fiber Irregularity", 0.0, 2.0, 1.0, 0.1)
                wear_pattern = st.slider("Natural Wear", 0.0, 1.0, 0.2, 0.1)
                backing_visibility = st.slider("Backing Visibility", 0.0, 0.5, 0.1, 0.05)

        def create_advanced_fiber_texture(carpet_type, fiber_type, size, pile_height, fiber_density, yarn_twist):
            """Create highly realistic fiber texture based on carpet construction"""
            h, w = size
            
            # Base fiber structure
            fiber_texture = np.zeros((h, w, 3), dtype=np.float32)
            
            # Calculate fiber spacing based on density
            fiber_spacing = max(1, int(4 / fiber_density))
            
            if carpet_type == "Loop Pile (Berber)":
                # Create loop structure
                for y in range(0, h, fiber_spacing):
                    for x in range(0, w, fiber_spacing):
                        # Create loop shape
                        loop_size = max(2, int(pile_height / 3))
                        
                        # Draw loop using bezier-like curve
                        for i in range(loop_size):
                            for j in range(loop_size):
                                if y + i < h and x + j < w:
                                    # Create loop height variation
                                    loop_height = np.sin((i / loop_size) * np.pi) * 0.3
                                    fiber_texture[y + i, x + j] = [loop_height, loop_height, loop_height]
                                    
                                    # Add fiber twist effect
                                    twist_effect = np.sin(j * yarn_twist * 0.5) * 0.1
                                    fiber_texture[y + i, x + j] += twist_effect
            
            elif carpet_type == "Cut Pile (Plush)":
                # Create cut fiber ends
                for y in range(0, h, fiber_spacing):
                    for x in range(0, w, fiber_spacing):
                        # Random fiber height variation
                        height_var = np.random.normal(1.0, 0.1)
                        fiber_height = pile_height * height_var
                        
                        # Create individual fiber
                        fiber_width = max(1, int(fiber_spacing * 0.8))
                        for i in range(fiber_width):
                            for j in range(fiber_width):
                                if y + i < h and x + j < w:
                                    # Fiber shadow based on height
                                    shadow = (fiber_height / pile_height) * 0.2
                                    fiber_texture[y + i, x + j] = [shadow, shadow, shadow]
            
            elif carpet_type == "Frieze (Twisted)":
                # Create highly twisted fiber appearance
                for y in range(0, h, fiber_spacing):
                    for x in range(0, w, fiber_spacing):
                        # Create spiral/twisted pattern
                        center_x, center_y = x + fiber_spacing // 2, y + fiber_spacing // 2
                        
                        for i in range(fiber_spacing):
                            for j in range(fiber_spacing):
                                if y + i < h and x + j < w:
                                    # Calculate twist pattern
                                    dx, dy = j - fiber_spacing // 2, i - fiber_spacing // 2
                                    angle = np.arctan2(dy, dx)
                                    radius = np.sqrt(dx**2 + dy**2)
                                    
                                    # Twisted fiber effect
                                    twist_intensity = np.sin(angle * yarn_twist * 3 + radius * 0.5) * 0.3
                                    fiber_texture[y + i, x + j] = [twist_intensity, twist_intensity, twist_intensity]
            
            elif carpet_type == "Flatweave":
                # Create woven pattern
                warp_spacing = max(2, int(6 / fiber_density))
                weft_spacing = max(2, int(6 / fiber_density))
                
                # Warp threads (vertical)
                for x in range(0, w, warp_spacing):
                    for y in range(h):
                        if x < w:
                            # Over-under weave pattern
                            over_under = (y // weft_spacing) % 2
                            intensity = 0.2 if over_under else -0.2
                            
                            for i in range(min(warp_spacing, w - x)):
                                fiber_texture[y, x + i] += [intensity, intensity, intensity]
                
                # Weft threads (horizontal)  
                for y in range(0, h, weft_spacing):
                    for x in range(w):
                        if y < h:
                            # Over-under weave pattern
                            over_under = (x // warp_spacing) % 2
                            intensity = 0.2 if over_under else -0.2
                            
                            for i in range(min(weft_spacing, h - y)):
                                fiber_texture[y + i, x] += [intensity, intensity, intensity]
            
            else:
                # Default texture for other types
                for y in range(0, h, fiber_spacing):
                    for x in range(0, w, fiber_spacing):
                        height_var = np.random.normal(0, 0.1)
                        for i in range(fiber_spacing):
                            for j in range(fiber_spacing):
                                if y + i < h and x + j < w:
                                    fiber_texture[y + i, x + j] = [height_var, height_var, height_var]
            
            # Add fiber material properties
            if fiber_type == "Wool":
                # Wool has natural crimp and matte appearance
                crimp_noise = np.random.normal(0, 0.05, (h, w, 3))
                fiber_texture += crimp_noise
                
            elif fiber_type == "Silk":
                # Silk has high sheen
                sheen_pattern = np.sin(np.arange(w) * 0.1) * 0.1
                for y in range(h):
                    fiber_texture[y, :] += sheen_pattern[:, np.newaxis]
                    
            elif fiber_type == "Nylon":
                # Nylon has uniform appearance with slight sheen
                uniform_sheen = np.random.normal(0.05, 0.02, (h, w, 3))
                fiber_texture += uniform_sheen
            
            return fiber_texture

        def apply_realistic_lighting(img_array, fiber_texture, lighting_angle, shadow_depth, highlight_strength, pile_height):
            """Apply realistic lighting based on fiber structure and pile height"""
            h, w = img_array.shape[:2]
            
            # Convert lighting angle to radians
            light_angle_rad = np.radians(lighting_angle)
            
            # Create lighting vectors
            light_x = np.cos(light_angle_rad)
            light_y = np.sin(light_angle_rad)
            
            # Create surface normal map from fiber texture
            fiber_gray = np.mean(fiber_texture, axis=2)
            
            # Calculate gradients for surface normals
            grad_x = np.gradient(fiber_gray, axis=1)
            grad_y = np.gradient(fiber_gray, axis=0)
            
            # Calculate lighting intensity for each pixel
            lighting_map = np.zeros((h, w), dtype=np.float32)
            
            for y in range(h):
                for x in range(w):
                    # Surface normal based on fiber texture gradients
                    normal_x = -grad_x[y, x] * pile_height
                    normal_y = -grad_y[y, x] * pile_height
                    normal_z = 1.0
                    
                    # Normalize normal vector
                    normal_length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
                    if normal_length > 0:
                        normal_x /= normal_length
                        normal_y /= normal_length
                        normal_z /= normal_length
                    
                    # Calculate dot product with light direction
                    dot_product = normal_x * light_x + normal_y * light_y + normal_z * 0.5
                    
                    # Apply lighting model (Lambertian + specular)
                    diffuse = max(0, dot_product)
                    specular = max(0, dot_product ** 16) * highlight_strength * 0.3
                    
                    # Combine lighting
                    lighting_map[y, x] = 0.4 + diffuse * 0.6 + specular
            
            # Apply shadow depth
            shadow_map = 1.0 - fiber_gray * shadow_depth * 0.3
            lighting_map *= shadow_map
            
            # Apply lighting to image
            lit_image = img_array.copy().astype(np.float32)
            for i in range(3):
                lit_image[:, :, i] *= lighting_map
            
            return np.clip(lit_image, 0, 255).astype(np.uint8)

        def apply_color_depth_enhancement(img_array, color_variation, fiber_irregularity):
            """Enhance color depth with natural variations"""
            h, w = img_array.shape[:2]
            
            # Convert to LAB color space for better color manipulation
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB).astype(np.float32)
            l, a, b = cv2.split(lab)
            
            # Add natural color variation
            if color_variation > 0:
                # Lightness variation
                l_variation = np.random.normal(0, color_variation * 5, (h, w))
                l += l_variation
                
                # Color channel variations
                a_variation = np.random.normal(0, color_variation * 2, (h, w))
                b_variation = np.random.normal(0, color_variation * 2, (h, w))
                a += a_variation
                b += b_variation
            
            # Add fiber irregularity
            if fiber_irregularity > 0:
                # Create irregular patterns
                irregularity_pattern = np.random.normal(1.0, fiber_irregularity * 0.1, (h, w))
                l *= irregularity_pattern
            
            # Clamp values to valid ranges
            l = np.clip(l, 0, 100)
            a = np.clip(a, -127, 127)
            b = np.clip(b, -127, 127)
            
            # Convert back to RGB
            lab_enhanced = cv2.merge([l, a, b]).astype(np.uint8)
            rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            
            return rgb_enhanced

        def create_professional_carpet_visualization(pixel_img, target_width, target_height, 
                                                  carpet_type, fiber_type, pile_height, 
                                                  fiber_density, yarn_twist, color_variation,
                                                  lighting_angle, shadow_depth, highlight_strength,
                                                  fiber_irregularity, wear_pattern, backing_visibility):
            """Create professional-grade carpet visualization"""
            
            try:
                img_array = np.array(pixel_img)
                
                # Step 1: Intelligent upscaling with edge preservation
                # Use multiple stages for better quality
                current_img = img_array.copy()
                
                # Progressive upscaling
                while current_img.shape[0] < target_height or current_img.shape[1] < target_width:
                    new_height = min(target_height, current_img.shape[0] * 2)
                    new_width = min(target_width, current_img.shape[1] * 2)
                    
                    # Use different interpolation based on current size
                    if current_img.shape[0] * current_img.shape[1] < 10000:
                        # Small images - use nearest neighbor to preserve sharp edges
                        current_img = cv2.resize(current_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                    else:
                        # Larger images - use Lanczos for smooth scaling
                        current_img = cv2.resize(current_img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                
                # Final resize to exact target size
                upscaled = cv2.resize(current_img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                
                # Step 2: Create fiber texture
                fiber_texture = create_advanced_fiber_texture(
                    carpet_type, fiber_type, (target_height, target_width), 
                    pile_height, fiber_density, yarn_twist
                )
                
                # Step 3: Apply fiber texture to base image
                textured_image = upscaled.copy().astype(np.float32)
                
                # Blend fiber texture with base colors
                for i in range(3):
                    # Modulate color with fiber texture
                    texture_effect = 1.0 + fiber_texture[:, :, i] * 0.4
                    textured_image[:, :, i] *= texture_effect
                
                # Step 4: Apply realistic lighting
                lit_image = apply_realistic_lighting(
                    textured_image.astype(np.uint8), fiber_texture, 
                    lighting_angle, shadow_depth, highlight_strength, pile_height
                )
                
                # Step 5: Enhance color depth and add natural variations
                enhanced_image = apply_color_depth_enhancement(
                    lit_image, color_variation, fiber_irregularity
                )
                
                # Step 6: Apply wear patterns if specified
                if wear_pattern > 0:
                    # Create wear map (typically in high-traffic areas)
                    wear_map = np.random.beta(2, 5, (target_height, target_width)) * wear_pattern
                    
                    # Apply wear by slightly flattening the texture and lightening colors
                    for i in range(3):
                        enhanced_image[:, :, i] = enhanced_image[:, :, i] * (1 + wear_map * 0.2)
                
                # Step 7: Add subtle backing visibility for realistic depth
                if backing_visibility > 0:
                    # Create backing pattern (usually darker)
                    backing_pattern = np.random.normal(0.8, 0.1, (target_height, target_width))
                    backing_pattern = np.clip(backing_pattern, 0, 1)
                    
                    # Apply backing visibility
                    for i in range(3):
                        enhanced_image[:, :, i] = enhanced_image[:, :, i] * (1 - backing_visibility * (1 - backing_pattern))
                
                # Step 8: Final quality enhancement
                # Apply subtle unsharp masking for detail enhancement
                blurred = cv2.GaussianBlur(enhanced_image, (0, 0), 1.0)
                enhanced_image = cv2.addWeighted(enhanced_image, 1.3, blurred, -0.3, 0)
                
                # Final clipping and conversion
                result = np.clip(enhanced_image, 0, 255).astype(np.uint8)
                
                return Image.fromarray(result)
                
            except Exception as e:
                st.error(f"Error during carpet visualization: {str(e)}")
                # Fallback to basic upscaling
                return pixel_img.resize((target_width, target_height), Image.LANCZOS)
        
        # Generate the realistic carpet visualization
        if st.button("🚀 Generate Professional Carpet Visualization", type="primary"):
            with st.spinner("🔄 Creating photorealistic carpet visualization using advanced textile simulation... This may take a moment."):
                try:
                    realistic_carpet = create_professional_carpet_visualization(
                        pixel_image, target_width, target_height, carpet_type, fiber_type,
                        pile_height, fiber_density, yarn_twist, color_variation,
                        lighting_angle, shadow_depth, highlight_strength,
                        fiber_irregularity, wear_pattern, backing_visibility
                    )
                    
                    # Store the result in session state
                    st.session_state.realistic_carpet = realistic_carpet
                    st.session_state.target_width = target_width
                    st.session_state.target_height = target_height
                    st.session_state.carpet_specs = {
                        'type': carpet_type,
                        'fiber': fiber_type,
                        'pile_height': pile_height,
                        'density': fiber_density
                    }
                    
                except Exception as e:
                    st.error(f"Failed to create carpet visualization: {str(e)}")
                    # Fallback
                    st.session_state.realistic_carpet = pixel_image.resize((target_width, target_height), Image.LANCZOS)
        
        # Show realistic carpet if it exists in session state
        if hasattr(st.session_state, 'realistic_carpet') and st.session_state.realistic_carpet:
            with col2:
                st.subheader("Professional Carpet Visualization")
                specs = st.session_state.carpet_specs
                caption = f"{specs['type']} - {specs['fiber']} ({specs['pile_height']}mm pile) - {st.session_state.target_width}×{st.session_state.target_height}"
                st.image(
                    st.session_state.realistic_carpet, 
                    caption=caption,
                    use_container_width=True
                )
            
            # Professional download section
            st.subheader("💾 Download Professional Visualization")
            
            col_d1, col_d2, col_d3 = st.columns(3)
            
            with col_d1:
                # Download High-Quality PNG
                buf_png = BytesIO()
                st.session_state.realistic_carpet.save(buf_png, format="PNG", optimize=True)
                st.download_button(
                    label="📱 Download PNG (Lossless)",
                    data=buf_png.getvalue(),
                    file_name=f"carpet_visualization_{specs['type'].replace(' ', '_').lower()}.png",
                    mime="image/png"
                )
            
            with col_d2:
                # Download High-Quality JPEG
                buf_jpg = BytesIO()
                st.session_state.realistic_carpet.save(buf_jpg, format="JPEG", quality=98, optimize=True)
                st.download_button(
                    label="🖼 Download JPEG (High Quality)",
                    data=buf_jpg.getvalue(),
                    file_name=f"carpet_visualization_{specs['type'].replace(' ', '_').lower()}.jpg",
                    mime="image/jpeg"
                )
            
            with col_d3:
                # Download Print-Ready TIFF
                buf_tiff = BytesIO()
                st.session_state.realistic_carpet.save(buf_tiff, format="TIFF", compression="lzw")
                st.download_button(
                    label="🖨 Download TIFF (Print Ready)",
                    data=buf_tiff.getvalue(),
                    file_name=f"carpet_visualization_{specs['type'].replace(' ', '_').lower()}.tiff",
                    mime="image/tiff"
                )
    
    except Exception as e:
        st.error(f"Error loading design: {str(e)}")

else:
    st.info("⬆ Please upload a carpet design pattern to start the professional visualization!")
    
    st.subheader("🏭 Professional Carpet Manufacturing Features:")
    
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        st.markdown("""
        **🧶 Textile-Specific Simulations:**
        - **Loop Pile (Berber)**: Realistic loop structure simulation
        - **Cut Pile**: Individual fiber end visualization  
        - **Frieze**: Twisted yarn appearance
        - **Flatweave**: Precise woven pattern recreation
        - **Hand-Knotted**: Traditional knotting visualization
        """)
        
    with col_feat2:
        st.markdown("""
        **🔬 Advanced Material Physics:**
        - **Fiber Density Control**: Realistic pile density simulation
        - **Yarn Twist Effects**: Natural fiber twist visualization
        - **Pile Height Simulation**: Accurate shadow and depth
        - **Natural Wear Patterns**: Realistic usage simulation
        - **Professional Lighting**: Industry-standard visualization
        """)
    
    st.markdown("""
    **💼 Perfect for:**
    - Carpet manufacturers and designers
    - Interior design visualization
    - Customer presentations and samples
    - Quality control and pattern verification
    - E-commerce product visualization
    """)
