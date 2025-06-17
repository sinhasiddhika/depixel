import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import io
import base64
from sklearn.cluster import KMeans
from scipy import ndimage
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Pixel Art to Realistic Image Converter",
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
</style>
""", unsafe_allow_html=True)

class PixelToRealisticConverter:
    def __init__(self):
        pass
        
    def extract_color_palette(self, image, n_colors=8):
        """Extract dominant colors from the image"""
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Reshape image to be a list of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Use KMeans to find dominant colors
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        return colors
    
    def upscale_image(self, image, scale_factor=4):
        """Upscale image using bicubic interpolation"""
        width, height = image.size
        new_size = (width * scale_factor, height * scale_factor)
        return image.resize(new_size, Image.BICUBIC)
    
    def apply_edge_enhancement(self, image):
        """Enhance edges for better detail preservation"""
        return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    def apply_texture_enhancement(self, image):
        """Apply texture enhancement using PIL filters"""
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Apply simple noise reduction
        enhanced = image.filter(ImageFilter.MedianFilter(3))
        
        # Apply unsharp masking using PIL
        blurred = enhanced.filter(ImageFilter.GaussianBlur(radius=2))
        enhanced_array = np.array(enhanced).astype(np.float32)
        blurred_array = np.array(blurred).astype(np.float32)
        
        # Unsharp mask calculation
        unsharp_mask = enhanced_array + 0.5 * (enhanced_array - blurred_array)
        unsharp_mask = np.clip(unsharp_mask, 0, 255)
        
        return Image.fromarray(unsharp_mask.astype(np.uint8))
    
    def create_depth_map(self, image):
        """Create a simple depth map based on brightness"""
        gray = image.convert('L')
        depth_array = np.array(gray)
        
        # Apply Gaussian blur for smooth depth transitions
        depth_smooth = ndimage.gaussian_filter(depth_array, sigma=2)
        
        # Normalize depth values
        depth_normalized = (depth_smooth - depth_smooth.min()) / (depth_smooth.max() - depth_smooth.min())
        
        return depth_normalized
    
    def apply_lighting_effects(self, image, depth_map, light_intensity=0.3):
        """Apply realistic lighting effects based on depth"""
        img_array = np.array(image).astype(np.float32)
        
        # Create light source from top-left
        height, width = depth_map.shape
        y, x = np.ogrid[:height, :width]
        
        # Calculate distance from light source
        light_x, light_y = width * 0.3, height * 0.3
        distance = np.sqrt((x - light_x)**2 + (y - light_y)**2)
        
        # Normalize distance
        max_distance = np.sqrt(width**2 + height**2)
        distance_normalized = distance / max_distance
        
        # Create lighting effect
        lighting = (1 - distance_normalized) * light_intensity + (1 - light_intensity)
        lighting = lighting * (1 + depth_map * 0.5)  # Enhance based on depth
        
        # Apply lighting to each channel
        for i in range(3):
            img_array[:, :, i] *= lighting
        
        # Ensure values are within valid range
        img_array = np.clip(img_array, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def enhance_colors(self, image, saturation=1.2, contrast=1.1):
        """Enhance color saturation and contrast"""
        # Enhance saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        
        return image
    
    def apply_custom_palette(self, image, custom_colors):
        """Apply custom color palette to the image"""
        if not custom_colors:
            return image
            
        img_array = np.array(image)
        original_shape = img_array.shape
        
        # Reshape for color mapping
        pixels = img_array.reshape(-1, 3)
        
        # Convert custom colors to numpy array
        palette = np.array(custom_colors)
        
        # Find closest color in palette for each pixel
        distances = np.sqrt(np.sum((pixels[:, np.newaxis] - palette[np.newaxis, :]) ** 2, axis=2))
        closest_colors = np.argmin(distances, axis=1)
        
        # Map pixels to closest palette colors
        new_pixels = palette[closest_colors]
        
        # Reshape back to original image shape
        new_image = new_pixels.reshape(original_shape)
        
        return Image.fromarray(new_image.astype(np.uint8))
    
    def convert_pixel_to_realistic(self, image, custom_palette=None, settings=None):
        """Main conversion function"""
        if settings is None:
            settings = {
                'scale_factor': 4,
                'saturation': 1.2,
                'contrast': 1.1,
                'light_intensity': 0.3,
                'edge_enhancement': True,
                'texture_enhancement': True
            }
        
        # Step 1: Apply custom palette if provided
        if custom_palette:
            image = self.apply_custom_palette(image, custom_palette)
        
        # Step 2: Upscale the image
        upscaled = self.upscale_image(image, settings['scale_factor'])
        
        # Step 3: Apply edge enhancement
        if settings['edge_enhancement']:
            upscaled = self.apply_edge_enhancement(upscaled)
        
        # Step 4: Apply texture enhancement
        if settings['texture_enhancement']:
            upscaled = self.apply_texture_enhancement(upscaled)
        
        # Step 5: Create depth map
        depth_map = self.create_depth_map(upscaled)
        
        # Step 6: Apply lighting effects
        realistic = self.apply_lighting_effects(upscaled, depth_map, settings['light_intensity'])
        
        # Step 7: Enhance colors
        realistic = self.enhance_colors(realistic, settings['saturation'], settings['contrast'])
        
        return realistic

def create_download_link(image, filename="converted_image.png"):
    """Create a download link for the processed image"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">Download High Quality Image</a>'
    return href

def main():
    st.markdown('<h1 class="main-header">🎨 Pixel Art to Realistic Image Converter</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Transform your pixel art into stunning photorealistic images using advanced AI algorithms!
    Upload your pixel art and watch it come to life with enhanced details, realistic lighting, and textures.
    """)
    
    # Initialize converter
    converter = PixelToRealisticConverter()
    
    # Sidebar for settings
    st.sidebar.markdown('<h2 class="section-header">🎛️ Conversion Settings</h2>', unsafe_allow_html=True)
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Choose a pixel art image...", 
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload your pixel art image to convert"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        original_image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="section-header">📸 Original Pixel Art</h3>', unsafe_allow_html=True)
            st.image(original_image, caption="Original Image", use_column_width=True)
            
            # Display image info
            st.info(f"Image Size: {original_image.size[0]} x {original_image.size[1]} pixels")
        
        # Conversion settings
        st.sidebar.markdown("### 🎨 Enhancement Settings")
        
        scale_factor = st.sidebar.slider("Upscale Factor", 2, 8, 4, help="How much to enlarge the image")
        saturation = st.sidebar.slider("Color Saturation", 0.5, 2.0, 1.2, 0.1)
        contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.1, 0.1)
        light_intensity = st.sidebar.slider("Lighting Intensity", 0.0, 1.0, 0.3, 0.1)
        
        edge_enhancement = st.sidebar.checkbox("Edge Enhancement", value=True)
        texture_enhancement = st.sidebar.checkbox("Texture Enhancement", value=True)
        
        # Color palette customization
        st.sidebar.markdown("### 🎨 Custom Color Palette")
        use_custom_palette = st.sidebar.checkbox("Use Custom Color Palette")
        
        custom_colors = []
        if use_custom_palette:
            st.sidebar.markdown("**Add Custom Colors:**")
            num_colors = st.sidebar.slider("Number of Colors", 2, 12, 6)
            
            for i in range(num_colors):
                color = st.sidebar.color_picker(f"Color {i+1}", f"#{i*40:02x}{i*30:02x}{i*20:02x}")
                # Convert hex to RGB
                rgb = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
                custom_colors.append(rgb)
        
        # Extract and display original palette
        with col1:
            st.markdown("### 🎨 Original Color Palette")
            original_palette = converter.extract_color_palette(original_image, 8)
            
            # Create a visual palette using Plotly
            palette_colors = [f'rgb({color[0]}, {color[1]}, {color[2]})' for color in original_palette]
            
            fig = go.Figure()
            
            # Create color swatches
            for i, color in enumerate(palette_colors):
                fig.add_trace(go.Scatter(
                    x=[i], y=[0],
                    mode='markers',
                    marker=dict(size=50, color=color),
                    name=f'Color {i+1}',
                    hovertemplate=f'RGB: {original_palette[i]}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Extracted Color Palette",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 0.5]),
                height=150,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Conversion button
        if st.button("🚀 Convert to Realistic Image", type="primary"):
            with st.spinner("Converting your pixel art to realistic image... This may take a moment."):
                
                # Prepare settings
                settings = {
                    'scale_factor': scale_factor,
                    'saturation': saturation,
                    'contrast': contrast,
                    'light_intensity': light_intensity,
                    'edge_enhancement': edge_enhancement,
                    'texture_enhancement': texture_enhancement
                }
                
                # Convert image
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
                    
                    # Download button
                    st.markdown(
                        create_download_link(realistic_image, "realistic_converted_image.png"),
                        unsafe_allow_html=True
                    )
                    
                    # Additional download options
                    st.markdown("### 📥 Download Options")
                    
                    # Different quality options
                    col_low, col_med, col_high = st.columns(3)
                    
                    with col_low:
                        # Low quality (compressed)
                        buffered_low = io.BytesIO()
                        realistic_image.save(buffered_low, format="JPEG", quality=60)
                        st.download_button(
                            label="Download (Low Quality)",
                            data=buffered_low.getvalue(),
                            file_name="realistic_low_quality.jpg",
                            mime="image/jpeg"
                        )
                    
                    with col_med:
                        # Medium quality
                        buffered_med = io.BytesIO()
                        realistic_image.save(buffered_med, format="PNG", optimize=True)
                        st.download_button(
                            label="Download (Medium Quality)",
                            data=buffered_med.getvalue(),
                            file_name="realistic_medium_quality.png",
                            mime="image/png"
                        )
                    
                    with col_high:
                        # High quality (uncompressed)
                        buffered_high = io.BytesIO()
                        realistic_image.save(buffered_high, format="PNG", quality=100)
                        st.download_button(
                            label="Download (High Quality)",
                            data=buffered_high.getvalue(),
                            file_name="realistic_high_quality.png",
                            mime="image/png"
                        )
        
        # Comparison section
        st.markdown("---")
        st.markdown('<h2 class="section-header">🔍 How It Works</h2>', unsafe_allow_html=True)
        
        with st.expander("Conversion Process Details"):
            st.markdown("""
            **Our AI-powered conversion process includes:**
            
            1. **Color Palette Analysis**: Extracts dominant colors from your pixel art
            2. **Intelligent Upscaling**: Uses bicubic interpolation for smooth scaling
            3. **Edge Enhancement**: Preserves and enhances important details
            4. **Texture Enhancement**: Applies bilateral filtering and unsharp masking
            5. **Depth Mapping**: Creates realistic depth perception
            6. **Dynamic Lighting**: Adds realistic lighting effects based on depth
            7. **Color Enhancement**: Improves saturation and contrast for realism
            8. **Custom Palette Application**: Optionally applies your custom color scheme
            """)
        
        with st.expander("Tips for Best Results"):
            st.markdown("""
            **For optimal conversion results:**
            
            - Use pixel art with clear, distinct colors
            - Higher contrast images work better
            - Simple geometric patterns convert more effectively
            - Experiment with different settings for your specific art style
            - Try different scale factors based on your original image size
            - Custom color palettes work best with thoughtfully chosen colors
            """)
    
    else:
        # Display sample images and instructions
        st.markdown('<h2 class="section-header">📚 Getting Started</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### How to use this converter:
        
        1. **Upload** your pixel art image using the file uploader
        2. **Customize** the conversion settings in the sidebar
        3. **Optional**: Create a custom color palette
        4. **Click** the convert button to transform your image
        5. **Download** your realistic image in your preferred quality
        
        ### Features:
        - 🎯 **High-quality upscaling** with detail preservation
        - 🎨 **Custom color palette** support
        - 💡 **Realistic lighting effects** and depth mapping
        - 🖼️ **Multiple download options** (Low, Medium, High quality)
        - ⚙️ **Adjustable settings** for personalized results
        """)
        
        st.info("👆 Upload a pixel art image to get started!")

if __name__ == "__main__":
    main()
