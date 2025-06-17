import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import io
import base64
import math
import random
from typing import List, Tuple, Dict
import time

# Set page config
st.set_page_config(
    page_title="🧶 Realistic Carpet Pattern Generator",
    page_icon="🧶",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitCarpetGenerator:
    def __init__(self):
        self.color_presets = {
            'Classic Gray': ['#2c3e50', '#ecf0f1', '#34495e', '#95a5a6'],
            'Warm Earth': ['#8b4513', '#deb887', '#a0522d', '#f4a460'],
            'Cool Blue': ['#1e3a8a', '#e0f2fe', '#3b82f6', '#93c5fd'],
            'Luxury Gold': ['#8b4513', '#ffd700', '#a0522d', '#ffed4e'],
            'Persian Red': ['#8b0000', '#f5deb3', '#a0522d', '#cd853f'],
            'Moroccan Teal': ['#008080', '#f0ffff', '#4682b4', '#20b2aa'],
            'Desert Sand': ['#c19a6b', '#f5f5dc', '#daa520', '#cd853f'],
            'Forest Green': ['#228b22', '#f0fff0', '#32cd32', '#90ee90'],
            'Royal Purple': ['#4b0082', '#e6e6fa', '#9370db', '#dda0dd'],
            'Sunset Orange': ['#ff4500', '#fff8dc', '#ffa500', '#ffb347']
        }
        
        self.pattern_descriptions = {
            'Diamond': 'Classic geometric diamond pattern - perfect for modern and traditional spaces',
            'Chevron': 'Zigzag chevron pattern - adds dynamic energy to any room',
            'Persian': 'Traditional Persian mandala style - elegant and sophisticated',
            'Moroccan': 'Intricate Moroccan tile pattern - exotic and luxurious',
            'Tribal': 'Bold tribal/Aztec pattern - perfect for bohemian interiors',
            'Hexagonal': 'Modern hexagonal honeycomb pattern - contemporary geometric design'
        }
        
        self.material_info = {
            'Wool': 'Natural, durable, soft texture with excellent stain resistance',
            'Cotton': 'Smooth, breathable, easy to clean, perfect for high-traffic areas',
            'Jute': 'Eco-friendly, rustic texture, natural and sustainable',
            'Silk': 'Luxurious, elegant sheen, premium quality with fine details',
            'Synthetic': 'Affordable, consistent texture, stain-resistant and durable'
        }

    @st.cache_data
    def hex_to_rgb(_self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def create_base_canvas(self, width: int, height: int, base_color: str) -> Image.Image:
        """Create base canvas with background color"""
        rgb_color = self.hex_to_rgb(base_color)
        img = Image.new('RGB', (width, height), rgb_color)
        return img

    def generate_diamond_pattern(self, width: int, height: int, scale: int, colors: List[str]) -> Image.Image:
        """Generate diamond/rhombus geometric pattern"""
        img = self.create_base_canvas(width, height, colors[0])
        draw = ImageDraw.Draw(img)
        
        rows = height // scale + 2
        cols = width // scale + 2
        
        for row in range(rows):
            for col in range(cols):
                x = col * scale - scale // 2
                y = row * scale - scale // 2
                
                diamond_points = [
                    (x + scale // 2, y),
                    (x + scale, y + scale // 2),
                    (x + scale // 2, y + scale),
                    (x, y + scale // 2)
                ]
                
                color_idx = (row + col) % len(colors)
                fill_color = self.hex_to_rgb(colors[color_idx])
                
                draw.polygon(diamond_points, fill=fill_color)
                
                if len(colors) > 2:
                    border_color = self.hex_to_rgb(colors[(color_idx + 2) % len(colors)])
                    draw.polygon(diamond_points, outline=border_color, width=1)
        
        return img

    def generate_chevron_pattern(self, width: int, height: int, scale: int, colors: List[str]) -> Image.Image:
        """Generate chevron zigzag pattern"""
        img = self.create_base_canvas(width, height, colors[0])
        draw = ImageDraw.Draw(img)
        
        stripe_height = scale
        rows = height // stripe_height + 2
        
        for i in range(rows):
            y = i * stripe_height - stripe_height // 2
            color_idx = i % len(colors)
            fill_color = self.hex_to_rgb(colors[color_idx])
            
            chevron_points = [
                (0, y),
                (width // 2, y + stripe_height // 2),
                (width, y),
                (width, y + stripe_height),
                (width // 2, y + stripe_height // 2),
                (0, y + stripe_height)
            ]
            
            draw.polygon(chevron_points, fill=fill_color)
        
        return img

    def generate_persian_pattern(self, width: int, height: int, scale: int, colors: List[str]) -> Image.Image:
        """Generate Persian-style mandala pattern"""
        img = self.create_base_canvas(width, height, colors[0])
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = width // 2, height // 2
        
        for radius in range(scale, min(width, height) // 2, scale):
            petals = max(6, radius // scale * 4)
            
            for i in range(petals):
                angle = (i / petals) * 2 * math.pi
                x = center_x + math.cos(angle) * radius
                y = center_y + math.sin(angle) * radius
                
                color_idx = i % len(colors)
                fill_color = self.hex_to_rgb(colors[color_idx])
                
                petal_size = scale // 3
                bbox = [x - petal_size, y - petal_size, x + petal_size, y + petal_size]
                draw.ellipse(bbox, fill=fill_color)
        
        return img

    def generate_moroccan_pattern(self, width: int, height: int, scale: int, colors: List[str]) -> Image.Image:
        """Generate Moroccan tile pattern"""
        img = self.create_base_canvas(width, height, colors[0])
        draw = ImageDraw.Draw(img)
        
        for x in range(0, width, scale):
            for y in range(0, height, scale):
                center_x = x + scale // 2
                center_y = y + scale // 2
                
                # Draw star pattern
                star_points = []
                outer_radius = scale // 3
                inner_radius = scale // 6
                points = 8
                
                for i in range(points * 2):
                    angle = (i / (points * 2)) * 2 * math.pi
                    radius = outer_radius if i % 2 == 0 else inner_radius
                    px = center_x + math.cos(angle) * radius
                    py = center_y + math.sin(angle) * radius
                    star_points.append((px, py))
                
                color_idx = ((x // scale) + (y // scale)) % len(colors)
                fill_color = self.hex_to_rgb(colors[color_idx])
                
                draw.polygon(star_points, fill=fill_color)
                
                # Add circular border
                circle_radius = scale // 2.5
                bbox = [center_x - circle_radius, center_y - circle_radius,
                       center_x + circle_radius, center_y + circle_radius]
                if len(colors) > 1:
                    border_color = self.hex_to_rgb(colors[(color_idx + 1) % len(colors)])
                    draw.ellipse(bbox, outline=border_color, width=2)
        
        return img

    def generate_tribal_pattern(self, width: int, height: int, scale: int, colors: List[str]) -> Image.Image:
        """Generate tribal/aztec pattern"""
        img = self.create_base_canvas(width, height, colors[0])
        draw = ImageDraw.Draw(img)
        
        random.seed(42)  # For consistent results
        
        for x in range(0, width, scale):
            for y in range(0, height, scale):
                color_idx = random.randint(0, len(colors) - 1)
                fill_color = self.hex_to_rgb(colors[color_idx])
                draw.rectangle([x, y, x + scale, y + scale], fill=fill_color)
                
                # Inner pattern
                inner_size = scale // 2
                inner_x = x + scale // 4
                inner_y = y + scale // 4
                
                inner_color_idx = (color_idx + 2) % len(colors)
                inner_color = self.hex_to_rgb(colors[inner_color_idx])
                draw.rectangle([inner_x, inner_y, inner_x + inner_size, inner_y + inner_size], 
                             fill=inner_color)
                
                # Add triangle details
                triangle_points = [
                    (inner_x + inner_size // 2, inner_y),
                    (inner_x, inner_y + inner_size),
                    (inner_x + inner_size, inner_y + inner_size)
                ]
                detail_color = self.hex_to_rgb(colors[(color_idx + 1) % len(colors)])
                draw.polygon(triangle_points, fill=detail_color)
        
        return img

    def generate_hexagonal_pattern(self, width: int, height: int, scale: int, colors: List[str]) -> Image.Image:
        """Generate hexagonal honeycomb pattern"""
        img = self.create_base_canvas(width, height, colors[0])
        draw = ImageDraw.Draw(img)
        
        hex_size = scale // 2
        hex_height = hex_size * math.sqrt(3)
        
        for row in range(-2, int(height / hex_height) + 3):
            for col in range(-2, int(width / (hex_size * 1.5)) + 3):
                x = col * hex_size * 1.5
                y = row * hex_height + (col % 2) * hex_height / 2
                
                # Hexagon vertices
                hex_points = []
                for i in range(6):
                    angle = i * math.pi / 3
                    px = x + hex_size * math.cos(angle)
                    py = y + hex_size * math.sin(angle)
                    hex_points.append((px, py))
                
                color_idx = (row + col) % len(colors)
                fill_color = self.hex_to_rgb(colors[color_idx])
                
                draw.polygon(hex_points, fill=fill_color)
                if len(colors) > 2:
                    border_color = self.hex_to_rgb(colors[(color_idx + 1) % len(colors)])
                    draw.polygon(hex_points, outline=border_color, width=1)
        
        return img

    def apply_material_texture(self, img: Image.Image, material: str, density: float = 0.7) -> Image.Image:
        """Apply realistic material texture"""
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        np.random.seed(42)  # For consistent results
        
        if material == 'Wool':
            noise = np.random.normal(0, 15 * density, img_array.shape)
            # Add fiber-like directional noise
            for i in range(0, height, 3):
                for j in range(0, width, 3):
                    direction = np.random.uniform(0, 2 * np.pi)
                    length = int(3 * density)
                    for k in range(length):
                        ni = int(i + k * np.cos(direction))
                        nj = int(j + k * np.sin(direction))
                        if 0 <= ni < height and 0 <= nj < width:
                            noise[ni, nj] += np.random.normal(0, 5)
                            
        elif material == 'Cotton':
            noise = np.random.normal(0, 8 * density, img_array.shape)
            
        elif material == 'Jute':
            noise = np.random.normal(0, 20 * density, img_array.shape)
            noise += np.random.choice([-10, 0, 10], size=img_array.shape, p=[0.2, 0.6, 0.2])
            
        elif material == 'Silk':
            noise = np.random.normal(0, 5 * density, img_array.shape)
            # Add sheen effect
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            sheen = 10 * np.sin((x + y) / 20) * density
            noise[:, :, 0] += sheen
            noise[:, :, 1] += sheen
            noise[:, :, 2] += sheen
            
        else:  # Synthetic
            noise = np.random.normal(0, 6 * density, img_array.shape)
        
        textured_array = img_array.astype(np.float64) + noise
        textured_array = np.clip(textured_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(textured_array)

    def add_realistic_effects(self, img: Image.Image, lighting_intensity: float = 1.0) -> Image.Image:
        """Add lighting, depth, and realistic effects"""
        width, height = img.size
        
        # Create lighting gradient
        gradient = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(gradient)
        
        center_x, center_y = width // 2, height // 2
        max_radius = math.sqrt(center_x**2 + center_y**2)
        
        for radius in range(0, int(max_radius), 5):
            alpha = int(30 * (radius / max_radius) * lighting_intensity)
            color = (0, 0, 0, alpha)
            bbox = [center_x - radius, center_y - radius, 
                   center_x + radius, center_y + radius]
            draw.ellipse(bbox, fill=color)
        
        # Apply effects
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, gradient)
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)
        
        # Add subtle blur for fiber softness
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return img.convert('RGB')

def get_image_download_link(img: Image.Image, filename: str) -> str:
    """Generate download link for image"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">📥 Download {filename}</a>'
    return href

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .pattern-card {
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #f0f2f6;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        margin: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">🧶 Realistic Carpet Pattern Generator</h1>', unsafe_allow_html=True)
    st.markdown("### Create photorealistic carpet designs with authentic textures and patterns")

    # Initialize generator
    generator = StreamlitCarpetGenerator()

    # Sidebar controls
    st.sidebar.markdown("## 🎨 Design Controls")
    
    # Pattern selection
    st.sidebar.markdown("### Pattern Type")
    pattern = st.sidebar.selectbox(
        "Choose Pattern Style",
        options=['Diamond', 'Chevron', 'Persian', 'Moroccan', 'Tribal', 'Hexagonal'],
        help="Select the geometric pattern for your carpet"
    )
    
    # Display pattern description
    st.sidebar.markdown(f"*{generator.pattern_descriptions[pattern]}*")
    
    # Material selection
    st.sidebar.markdown("### Material Type")
    material = st.sidebar.selectbox(
        "Choose Carpet Material",
        options=['Wool', 'Cotton', 'Jute', 'Silk', 'Synthetic'],
        help="Different materials create different textures"
    )
    
    # Display material info
    st.sidebar.markdown(f"*{generator.material_info[material]}*")
    
    # Size and scale controls
    st.sidebar.markdown("### Dimensions & Scale")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        width = st.number_input("Width", min_value=200, max_value=1000, value=500, step=50)
    with col2:
        height = st.number_input("Height", min_value=200, max_value=1000, value=500, step=50)
    
    scale = st.sidebar.slider("Pattern Scale", min_value=15, max_value=80, value=35, step=5,
                             help="Larger values create bigger pattern elements")
    
    # Texture controls
    st.sidebar.markdown("### Texture Settings")
    density = st.sidebar.slider("Fiber Density", min_value=0.1, max_value=1.0, value=0.7, step=0.1,
                               help="Higher values create more pronounced texture")
    
    lighting = st.sidebar.slider("Lighting Intensity", min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                                help="Adjust the depth and lighting effects")
    
    # Color selection
    st.sidebar.markdown("### Color Palette")
    
    use_preset = st.sidebar.checkbox("Use Color Preset", value=True)
    
    if use_preset:
        preset_name = st.sidebar.selectbox(
            "Choose Color Preset",
            options=list(generator.color_presets.keys()),
            index=0
        )
        colors = generator.color_presets[preset_name]
        
        # Display preset colors
        st.sidebar.markdown("**Preview Colors:**")
        cols = st.sidebar.columns(len(colors))
        for i, color in enumerate(colors):
            with cols[i]:
                st.color_picker(f"Color {i+1}", value=color, disabled=True, key=f"preset_{i}")
    else:
        st.sidebar.markdown("**Custom Colors:**")
        colors = []
        num_colors = st.sidebar.slider("Number of Colors", min_value=2, max_value=6, value=4)
        
        for i in range(num_colors):
            default_colors = ['#2c3e50', '#ecf0f1', '#34495e', '#95a5a6', '#7f8c8d', '#bdc3c7']
            default_color = default_colors[i] if i < len(default_colors) else '#000000'
            color = st.sidebar.color_picker(f"Color {i+1}", value=default_color, key=f"custom_{i}")
            colors.append(color)
    
    # Generate button
    st.sidebar.markdown("---")
    generate_button = st.sidebar.button("🎨 Generate Carpet", type="primary", use_container_width=True)
    
    # Main content area
    if generate_button or 'generated_carpet' not in st.session_state:
        with st.spinner("🧶 Weaving your carpet... This may take a moment!"):
            # Generate the carpet
            progress_bar = st.progress(0)
            
            # Step 1: Create base pattern
            progress_bar.progress(25)
            if pattern == 'Diamond':
                img = generator.generate_diamond_pattern(width, height, scale, colors)
            elif pattern == 'Chevron':
                img = generator.generate_chevron_pattern(width, height, scale, colors)
            elif pattern == 'Persian':
                img = generator.generate_persian_pattern(width, height, scale, colors)
            elif pattern == 'Moroccan':
                img = generator.generate_moroccan_pattern(width, height, scale, colors)
            elif pattern == 'Tribal':
                img = generator.generate_tribal_pattern(width, height, scale, colors)
            else:  # Hexagonal
                img = generator.generate_hexagonal_pattern(width, height, scale, colors)
            
            # Step 2: Apply material texture
            progress_bar.progress(50)
            img = generator.apply_material_texture(img, material, density)
            
            # Step 3: Add realistic effects
            progress_bar.progress(75)
            img = generator.add_realistic_effects(img, lighting)
            
            progress_bar.progress(100)
            time.sleep(0.5)  # Brief pause to show completion
            
            # Store in session state
            st.session_state.generated_carpet = img
            st.session_state.carpet_config = {
                'pattern': pattern,
                'material': material,
                'width': width,
                'height': height,
                'scale': scale,
                'density': density,
                'lighting': lighting,
                'colors': colors,
                'preset': preset_name if use_preset else 'Custom'
            }
            
            progress_bar.empty()
    
    # Display generated carpet
    if 'generated_carpet' in st.session_state:
        st.markdown("## 🏺 Your Generated Carpet")
        
        # Display carpet image
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(st.session_state.generated_carpet, caption="Generated Realistic Carpet Pattern", use_column_width=True)
        
        # Carpet specifications
        config = st.session_state.carpet_config
        
        st.markdown("### 📋 Carpet Specifications")
        
        spec_cols = st.columns(4)
        with spec_cols[0]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Pattern</h4>
                <p>{config['pattern']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with spec_cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Material</h4>
                <p>{config['material']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with spec_cols[2]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Dimensions</h4>
                <p>{config['width']}×{config['height']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with spec_cols[3]:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Color Scheme</h4>
                <p>{config['preset']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Download section
        st.markdown("### 💾 Download Options")
        
        download_cols = st.columns(3)
        
        with download_cols[0]:
            filename = f"{config['pattern'].lower()}_{config['material'].lower()}_carpet.png"
            st.markdown(get_image_download_link(st.session_state.generated_carpet, filename), 
                       unsafe_allow_html=True)
        
        with download_cols[1]:
            # High resolution version
            if st.button("🔍 Generate High-Res (800×800)"):
                with st.spinner("Creating high-resolution version..."):
                    high_res_img = generator.generate_diamond_pattern(800, 800, int(scale * 1.6), colors) if config['pattern'] == 'Diamond' else st.session_state.generated_carpet
                    high_res_img = generator.apply_material_texture(high_res_img, material, density)
                    high_res_img = generator.add_realistic_effects(high_res_img, lighting)
                    
                    st.session_state.high_res_carpet = high_res_img
        
        with download_cols[2]:
            if 'high_res_carpet' in st.session_state:
                high_res_filename = f"high_res_{filename}"
                st.markdown(get_image_download_link(st.session_state.high_res_carpet, high_res_filename), 
                           unsafe_allow_html=True)
        
        # Pattern variations
        st.markdown("### 🎨 Quick Style Variations")
        
        if st.button("Generate Style Variations"):
            st.markdown("#### Style Comparison")
            
            variation_cols = st.columns(3)
            
            # Variation 1: Different density
            with variation_cols[0]:
                var1 = generator.apply_material_texture(
                    st.session_state.generated_carpet.copy(), 
                    material, 
                    min(1.0, density + 0.3)
                )
                st.image(var1, caption="Higher Texture Density", use_column_width=True)
            
            # Variation 2: Different lighting
            with variation_cols[1]:
                var2 = generator.add_realistic_effects(
                    st.session_state.generated_carpet.copy(), 
                    max(0.1, lighting - 0.5)
                )
                st.image(var2, caption="Softer Lighting", use_column_width=True)
            
            # Variation 3: Different material
            with variation_cols[2]:
                materials = ['Wool', 'Cotton', 'Silk', 'Jute', 'Synthetic']
                alt_material = materials[(materials.index(material) + 1) % len(materials)]
                var3 = generator.apply_material_texture(
                    st.session_state.generated_carpet.copy(), 
                    alt_material, 
                    density
                )
                st.image(var3, caption=f"Alternative: {alt_material}", use_column_width=True)
    
    # Footer with tips
    st.markdown("---")
    st.markdown("### 💡 Pro Tips")
    
    tips_cols = st.columns(2)
    
    with tips_cols[0]:
        st.markdown("""
        <div class="info-box">
        <h4>🎨 Design Tips</h4>
        <ul>
        <li><strong>Scale:</strong> Larger scales work better for room-sized views</li>
        <li><strong>Colors:</strong> High contrast creates more defined patterns</li>
        <li><strong>Materials:</strong> Wool adds the most realistic texture</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tips_cols[1]:
        st.markdown("""
        <div class="info-box">
        <h4>🏠 Interior Design</h4>
        <ul>
        <li><strong>Diamond:</strong> Perfect for modern and traditional spaces</li>
        <li><strong>Persian:</strong> Adds elegance to formal rooms</li>
        <li><strong>Chevron:</strong> Creates dynamic energy in contemporary spaces</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
