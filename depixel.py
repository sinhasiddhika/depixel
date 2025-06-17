import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import io
import base64
from scipy import ndimage
import cv2

# Set page config
st.set_page_config(
    page_title="Enhanced Pixel Art to Realistic Image Converter",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        margin: 1rem 0;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .color-swatch {
        display: inline-block;
        width: 40px;
        height: 40px;
        margin: 5px;
        border: 2px solid #333;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedPixelToRealisticConverter:
    def __init__(self):
        pass
    
    def detect_pattern_type(self, image):
        """Detect if the image is textile/fabric pattern or digital pixel art"""
        img_array = np.array(image.convert('L'))
        
        # Calculate texture measures
        # High frequency content indicates textile
        laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
        
        # Edge density
        edges = cv2.Canny(img_array, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Texture complexity using local binary patterns simulation
        rows, cols = img_array.shape
        texture_complexity = 0
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = img_array[i, j]
                neighbors = [
                    img_array[i-1, j-1], img_array[i-1, j], img_array[i-1, j+1],
                    img_array[i, j+1], img_array[i+1, j+1], img_array[i+1, j],
                    img_array[i+1, j-1], img_array[i, j-1]
                ]
                pattern = sum(1 for n in neighbors if n > center)
                texture_complexity += pattern
        
        texture_complexity /= ((rows-2) * (cols-2))
        
        # Decision logic
        is_textile = (laplacian_var > 200 and edge_density > 0.15 and texture_complexity > 3) or edge_density > 0.25
        
        return "textile" if is_textile else "pixel_art"
        
    def extract_color_palette(self, image, n_colors=8):
        """Extract dominant colors from the image using improved clustering"""
        img_array = np.array(image)
        pixels = img_array.reshape(-1, 3)
        
        # Use k-means-like approach with numpy
        unique_colors, counts = np.unique(pixels.reshape(-1, pixels.shape[-1]), axis=0, return_counts=True)
        
        if len(unique_colors) <= n_colors:
            return unique_colors
        
        # Sort by frequency and take most common colors
        sorted_indices = np.argsort(counts)[::-1]
        return unique_colors[sorted_indices[:n_colors]]
    
    def create_textile_texture(self, image, fiber_strength=0.3):
        """Create realistic textile fiber texture"""
        img_array = np.array(image).astype(np.float32)
        height, width = img_array.shape[:2]
        
        # Create fiber noise patterns
        np.random.seed(42)  # For reproducible results
        
        # Horizontal fiber pattern
        h_fibers = np.random.normal(0, fiber_strength * 20, (height, width))
        h_fibers = ndimage.gaussian_filter(h_fibers, sigma=(0.5, 2))
        
        # Vertical fiber pattern
        v_fibers = np.random.normal(0, fiber_strength * 20, (height, width))
        v_fibers = ndimage.gaussian_filter(v_fibers, sigma=(2, 0.5))
        
        # Diagonal weave pattern
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        weave_pattern = np.sin(x * 0.3) * np.sin(y * 0.3) * fiber_strength * 15
        
        # Combine fiber patterns
        fiber_texture = h_fibers + v_fibers + weave_pattern
        
        # Apply to each color channel
        for i in range(3):
            img_array[:, :, i] += fiber_texture
        
        return np.clip(img_array, 0, 255).astype(np.uint8)
    
    def create_weave_pattern(self, image, weave_intensity=0.4):
        """Create realistic weave pattern for textiles"""
        img_array = np.array(image).astype(np.float32)
        height, width = img_array.shape[:2]
        
        # Create warp and weft patterns
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Warp threads (vertical)
        warp = np.sin(x * 0.4) * weave_intensity
        
        # Weft threads (horizontal)
        weft = np.sin(y * 0.4) * weave_intensity
        
        # Create over-under weave pattern
        weave_mask = (np.sin(x * 0.2) * np.sin(y * 0.2)) > 0
        weave_pattern = np.where(weave_mask, warp, weft)
        
        # Add subtle shadow effects
        shadow_pattern = weave_pattern * 0.3
        
        # Apply weave pattern
        for i in range(3):
            img_array[:, :, i] *= (1 + weave_pattern * 0.1)
            img_array[:, :, i] += shadow_pattern * 10
        
        return np.clip(img_array, 0, 255).astype(np.uint8)
    
    def enhance_textile_edges(self, image):
        """Enhance edges specifically for textile patterns"""
        # Convert to array
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply bilateral filter to reduce noise while preserving edges
        bilateral = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        # Enhance edges using unsharp masking
        gaussian = cv2.GaussianBlur(bilateral, (0, 0), 2.0)
        unsharp = cv2.addWeighted(bilateral, 1.5, gaussian, -0.5, 0)
        
        # Convert back to RGB
        result = cv2.cvtColor(unsharp, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result)
    
    def apply_advanced_lighting(self, image, depth_map, light_type="natural"):
        """Apply advanced lighting effects"""
        img_array = np.array(image).astype(np.float32)
        height, width = depth_map.shape
        
        if light_type == "natural":
            # Natural diffused lighting
            y, x = np.ogrid[:height, :width]
            
            # Main light source from top-left
            main_light_x, main_light_y = width * 0.2, height * 0.2
            main_distance = np.sqrt((x - main_light_x)**2 + (y - main_light_y)**2)
            max_distance = np.sqrt(width**2 + height**2)
            main_lighting = 1 - (main_distance / max_distance) * 0.3
            
            # Fill light from bottom-right
            fill_light_x, fill_light_y = width * 0.8, height * 0.8
            fill_distance = np.sqrt((x - fill_light_x)**2 + (y - fill_light_y)**2)
            fill_lighting = 1 - (fill_distance / max_distance) * 0.1
            
            # Combine lighting
            combined_lighting = (main_lighting * 0.7 + fill_lighting * 0.3)
            
            # Enhance based on depth
            lighting_effect = combined_lighting * (1 + depth_map * 0.2)
            
        else:  # dramatic lighting
            # Single strong light source
            y, x = np.ogrid[:height, :width]
            light_x, light_y = width * 0.3, height * 0.3
            distance = np.sqrt((x - light_x)**2 + (y - light_y)**2)
            max_distance = np.sqrt(width**2 + height**2)
            
            # Exponential falloff for dramatic effect
            lighting_effect = np.exp(-distance / (max_distance * 0.3))
            lighting_effect = lighting_effect * (1 + depth_map * 0.5)
        
        # Apply lighting to each channel
        for i in range(3):
            img_array[:, :, i] *= lighting_effect
        
        return np.clip(img_array, 0, 255).astype(np.uint8)
    
    def create_depth_map_advanced(self, image):
        """Create advanced depth map with edge awareness"""
        gray = np.array(image.convert('L')).astype(np.float32)
        
        # Use gradient-based depth estimation
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Combine brightness and gradient information
        brightness_depth = gray / 255.0
        gradient_depth = 1 - (gradient_magnitude / gradient_magnitude.max())
        
        # Weighted combination
        depth_map = brightness_depth * 0.7 + gradient_depth * 0.3
        
        # Smooth the depth map
        depth_smooth = ndimage.gaussian_filter(depth_map, sigma=1.5)
        
        # Normalize
        depth_normalized = (depth_smooth - depth_smooth.min()) / (depth_smooth.max() - depth_smooth.min())
        
        return depth_normalized
    
    def upscale_smart(self, image, scale_factor=4, method="lanczos"):
        """Smart upscaling with different methods"""
        width, height = image.size
        new_size = (width * scale_factor, height * scale_factor)
        
        if method == "lanczos":
            return image.resize(new_size, Image.LANCZOS)
        elif method == "bicubic":
            return image.resize(new_size, Image.BICUBIC)
        else:  # nearest for pixel art
            return image.resize(new_size, Image.NEAREST)
    
    def convert_pixel_to_realistic(self, image, custom_palette=None, settings=None):
        """Enhanced main conversion function"""
        if settings is None:
            settings = {
                'scale_factor': 4,
                'saturation': 1.2,
                'contrast': 1.1,
                'light_intensity': 0.3,
                'light_type': 'natural',
                'edge_enhancement': True,
                'texture_enhancement': True,
                'textile_mode': False,
                'fiber_strength': 0.3,
                'weave_intensity': 0.4,
                'upscale_method': 'lanczos'
            }
        
        # Detect pattern type if not specified
        if not settings.get('force_mode'):
            pattern_type = self.detect_pattern_type(image)
            settings['textile_mode'] = (pattern_type == "textile")
        
        # Step 1: Apply custom palette if provided
        if custom_palette:
            image = self.apply_custom_palette(image, custom_palette)
        
        # Step 2: Smart upscaling
        if settings['textile_mode']:
            upscaled = self.upscale_smart(image, settings['scale_factor'], 'lanczos')
        else:
            upscaled = self.upscale_smart(image, settings['scale_factor'], 'nearest')
        
        # Step 3: Apply appropriate enhancement
        if settings['textile_mode']:
            # Textile-specific processing
            if settings['texture_enhancement']:
                upscaled = self.enhance_textile_edges(upscaled)
                
                # Add fiber texture
                upscaled_array = self.create_textile_texture(upscaled, settings['fiber_strength'])
                upscaled = Image.fromarray(upscaled_array)
                
                # Add weave pattern
                upscaled_array = self.create_weave_pattern(upscaled, settings['weave_intensity'])
                upscaled = Image.fromarray(upscaled_array)
        else:
            # Digital pixel art processing
            if settings['edge_enhancement']:
                upscaled = upscaled.filter(ImageFilter.EDGE_ENHANCE_MORE)
            
            if settings['texture_enhancement']:
                upscaled = self.apply_texture_enhancement(upscaled)
        
        # Step 4: Create advanced depth map
        depth_map = self.create_depth_map_advanced(upscaled)
        
        # Step 5: Apply advanced lighting
        realistic_array = self.apply_advanced_lighting(upscaled, depth_map, settings['light_type'])
        realistic = Image.fromarray(realistic_array)
        
        # Step 6: Enhance colors
        realistic = self.enhance_colors(realistic, settings['saturation'], settings['contrast'])
        
        return realistic
    
    def apply_custom_palette(self, image, custom_colors):
        """Apply custom color palette to the image"""
        if not custom_colors:
            return image
            
        img_array = np.array(image)
        original_shape = img_array.shape
        pixels = img_array.reshape(-1, 3)
        palette = np.array(custom_colors)
        
        # Find closest color in palette for each pixel
        distances = np.sqrt(np.sum((pixels[:, np.newaxis] - palette[np.newaxis, :]) ** 2, axis=2))
        closest_colors = np.argmin(distances, axis=1)
        new_pixels = palette[closest_colors]
        new_image = new_pixels.reshape(original_shape)
        
        return Image.fromarray(new_image.astype(np.uint8))
    
    def apply_texture_enhancement(self, image):
        """Apply texture enhancement using PIL filters"""
        enhanced = image.filter(ImageFilter.MedianFilter(3))
        blurred = enhanced.filter(ImageFilter.GaussianBlur(radius=2))
        enhanced_array = np.array(enhanced).astype(np.float32)
        blurred_array = np.array(blurred).astype(np.float32)
        
        unsharp_mask = enhanced_array + 0.5 * (enhanced_array - blurred_array)
        unsharp_mask = np.clip(unsharp_mask, 0, 255)
        
        return Image.fromarray(unsharp_mask.astype(np.uint8))
    
    def enhance_colors(self, image, saturation=1.2, contrast=1.1):
        """Enhance color saturation and contrast"""
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        
        return image

def create_download_link(image, filename="converted_image.png"):
    """Create a download link for the processed image"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">Download High Quality Image</a>'
    return href

def create_color_palette_html(colors):
    """Create HTML for color palette display"""
    html = '<div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0;">'
    for i, color in enumerate(colors):
        rgb_str = f"rgb({color[0]}, {color[1]}, {color[2]})"
        html += f'''
        <div style="text-align: center;">
            <div style="width: 50px; height: 50px; background-color: {rgb_str}; 
                 border: 2px solid #333; border-radius: 8px; margin-bottom: 5px;"></div>
            <small>RGB({color[0]}, {color[1]}, {color[2]})</small>
        </div>
        '''
    html += '</div>'
    return html

def main():
    st.markdown('<h1 class="main-header">🎨 Enhanced Pixel Art to Realistic Image Converter</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Transform your pixel art and textile patterns into stunning photorealistic images! 
    This enhanced version automatically detects whether your image is digital pixel art or a textile pattern
    and applies the appropriate conversion algorithm.
    """)
    
    # Initialize converter
    converter = EnhancedPixelToRealisticConverter()
    
    # Sidebar for settings
    st.sidebar.markdown('<h2 class="section-header">🎛️ Enhanced Settings</h2>', unsafe_allow_html=True)
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload your pixel art or textile pattern image"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        original_image = Image.open(uploaded_file).convert('RGB')
        
        # Detect pattern type
        pattern_type = converter.detect_pattern_type(original_image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="section-header">📸 Original Image</h3>', unsafe_allow_html=True)
            st.image(original_image, caption="Original Image", use_column_width=True)
            
            # Display image info and detected type
            st.info(f"Image Size: {original_image.size[0]} x {original_image.size[1]} pixels")
            
            if pattern_type == "textile":
                st.success("🧶 Detected: Textile/Fabric Pattern")
            else:
                st.success("🎮 Detected: Digital Pixel Art")
        
        # Enhanced conversion settings
        st.sidebar.markdown("### 🎨 Conversion Mode")
        
        force_mode = st.sidebar.checkbox("Override Auto-Detection")
        if force_mode:
            manual_mode = st.sidebar.selectbox(
                "Select Mode",
                ["Digital Pixel Art", "Textile/Fabric Pattern"],
                index=0 if pattern_type == "pixel_art" else 1
            )
            textile_mode = (manual_mode == "Textile/Fabric Pattern")
        else:
            textile_mode = (pattern_type == "textile")
        
        st.sidebar.markdown("### 🎨 Enhancement Settings")
        
        scale_factor = st.sidebar.slider("Upscale Factor", 2, 8, 4)
        saturation = st.sidebar.slider("Color Saturation", 0.5, 2.0, 1.2, 0.1)
        contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.1, 0.1)
        
        # Lighting settings
        light_type = st.sidebar.selectbox("Lighting Type", ["natural", "dramatic"])
        light_intensity = st.sidebar.slider("Lighting Intensity", 0.0, 1.0, 0.3, 0.1)
        
        edge_enhancement = st.sidebar.checkbox("Edge Enhancement", value=True)
        texture_enhancement = st.sidebar.checkbox("Texture Enhancement", value=True)
        
        # Textile-specific settings
        if textile_mode:
            st.sidebar.markdown("### 🧶 Textile Settings")
            fiber_strength = st.sidebar.slider("Fiber Texture Strength", 0.0, 1.0, 0.3, 0.1)
            weave_intensity = st.sidebar.slider("Weave Pattern Intensity", 0.0, 1.0, 0.4, 0.1)
        
        # Color palette customization
        st.sidebar.markdown("### 🎨 Custom Color Palette")
        use_custom_palette = st.sidebar.checkbox("Use Custom Color Palette")
        
        custom_colors = []
        if use_custom_palette:
            st.sidebar.markdown("**Add Custom Colors:**")
            num_colors = st.sidebar.slider("Number of Colors", 2, 12, 6)
            
            for i in range(num_colors):
                color = st.sidebar.color_picker(f"Color {i+1}", f"#{i*40:02x}{i*30:02x}{i*20:02x}")
                rgb = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
                custom_colors.append(rgb)
        
        # Extract and display original palette
        with col1:
            st.markdown("### 🎨 Color Palette Analysis")
            original_palette = converter.extract_color_palette(original_image, 8)
            st.markdown(create_color_palette_html(original_palette), unsafe_allow_html=True)
        
        # Conversion button
        if st.button("🚀 Convert to Realistic Image", type="primary"):
            with st.spinner("Converting your image... This may take a moment."):
                
                # Prepare settings
                settings = {
                    'scale_factor': scale_factor,
                    'saturation': saturation,
                    'contrast': contrast,
                    'light_intensity': light_intensity,
                    'light_type': light_type,
                    'edge_enhancement': edge_enhancement,
                    'texture_enhancement': texture_enhancement,
                    'textile_mode': textile_mode,
                    'force_mode': force_mode
                }
                
                # Add textile-specific settings
                if textile_mode:
                    settings.update({
                        'fiber_strength': fiber_strength,
                        'weave_intensity': weave_intensity
                    })
                
                # Convert image
                try:
                    realistic_image = converter.convert_pixel_to_realistic(
                        original_image, 
                        custom_colors if use_custom_palette else None,
                        settings
                    )
                    
                    # Display result
                    with col2:
                        st.markdown('<h3 class="section-header">✨ Realistic Result</h3>', unsafe_allow_html=True)
                        st.image(realistic_image, caption="Converted Realistic Image", use_column_width=True)
                        
                        st.success(f"Conversion completed! New size: {realistic_image.size[0]} x {realistic_image.size[1]} pixels")
                        
                        # Download buttons
                        st.markdown("### 📥 Download Options")
                        
                        col_low, col_med, col_high = st.columns(3)
                        
                        with col_low:
                            buffered_low = io.BytesIO()
                            realistic_image.save(buffered_low, format="JPEG", quality=60)
                            st.download_button(
                                label="Download (Low Quality)",
                                data=buffered_low.getvalue(),
                                file_name="realistic_low_quality.jpg",
                                mime="image/jpeg"
                            )
                        
                        with col_med:
                            buffered_med = io.BytesIO()
                            realistic_image.save(buffered_med, format="PNG", optimize=True)
                            st.download_button(
                                label="Download (Medium Quality)",
                                data=buffered_med.getvalue(),
                                file_name="realistic_medium_quality.png",
                                mime="image/png"
                            )
                        
                        with col_high:
                            buffered_high = io.BytesIO()
                            realistic_image.save(buffered_high, format="PNG", quality=100)
                            st.download_button(
                                label="Download (High Quality)",
                                data=buffered_high.getvalue(),
                                file_name="realistic_high_quality.png",
                                mime="image/png"
                            )
                            
                except Exception as e:
                    st.error(f"Error during conversion: {str(e)}")
                    st.info("Try adjusting the settings or using a different image.")
        
        # Show custom palette preview if enabled
        if use_custom_palette and custom_colors:
            st.markdown("### 🎨 Your Custom Palette")
            st.markdown(create_color_palette_html(custom_colors), unsafe_allow_html=True)
        
        # Enhanced information section
        st.markdown("---")
        st.markdown('<h2 class="section-header">🔍 Enhanced Processing Details</h2>', unsafe_allow_html=True)
        
        with st.expander("New Features in Enhanced Version"):
            st.markdown(f"""
            **Automatic Pattern Detection**: {"🧶 Textile Pattern" if pattern_type == "textile" else "🎮 Digital Pixel Art"}
            
            **Enhanced Processing Pipeline:**
            
            **For Textile Patterns:**
            - Advanced fiber texture generation
            - Realistic weave pattern simulation  
            - Bilateral filtering for fabric-like smoothness
            - Natural lighting optimized for textiles
            
            **For Digital Pixel Art:**
            - Pixel-perfect upscaling with nearest neighbor
            - Sharp edge preservation
            - Retro-style enhancement filters
            - Dramatic lighting for digital aesthetics
            
            **Advanced Features:**
            - Gradient-based depth mapping
            - Multiple lighting scenarios (natural/dramatic)
            - Smart upscaling algorithm selection
            - Enhanced color palette analysis
            """)
        
        with st.expander("Tips for Best Results"):
            st.markdown("""
            **For Textile/Fabric Patterns:**
            - Use high-contrast woven patterns for best results
            - Adjust fiber strength and weave intensity for your specific fabric type
            - Natural lighting works best for realistic textile appearance
            
            **For Digital Pixel Art:**
            - Use images with clear, blocky pixel structure
            - Higher scale factors work well for small pixel art
            - Dramatic lighting enhances the digital aesthetic
            - Custom palettes can create unique artistic effects
            
            **General Tips:**
            - Let auto-detection choose the best processing mode
            - Experiment with different lighting types
            - Higher resolution source images produce better results
            """)
    
    else:
        # Display enhanced instructions
        st.markdown('<h2 class="section-header">🚀 Enhanced Features</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🧶 Textile Pattern Support
            - **Automatic Detection**: Recognizes fabric/textile patterns
            - **Fiber Texture Generation**: Creates realistic fiber appearance
            - **Weave Pattern Simulation**: Adds authentic weave structure
            - **Fabric-Optimized Lighting**: Natural lighting for textiles
            """)
        
        with col2:
            st.markdown("""
            ### 🎮 Enhanced Pixel Art Processing
            - **Smart Upscaling**: Preserves pixel-perfect edges
            - **Advanced Edge Enhancement**: Maintains sharp details
            - **Multiple Lighting Modes**: Natural and dramatic options
            - **Intelligent Color Processing**: Improved palette handling
            """)
        
        st.markdown("""
        ### How to use this enhanced converter:
        
        1. **Upload** your image (pixel art or textile pattern)
        2. **Auto-detection** will identify the pattern type
        3. **Customize** settings based on your image type
        4. **Convert** and download your enhanced realistic image
        
        ### New Capabilities:
        - 🔍 **Automatic pattern type detection**
        - 🧶 **Specialized textile processing algorithms**  
        - 💡 **Advanced lighting simulation**
        - 🎨 **Improved color enhancement**
        - ⚡ **Faster processing with better quality**
        """)
        
        st.info("👆 Upload an image to experience the enhanced conversion!")

if __name__ == "__main__":
    main()
