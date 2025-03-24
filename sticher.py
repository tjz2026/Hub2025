import numpy as np


def merge_patches(patches, patch_positions, patch_size, halo_size, output_shape, grid_type='cornered'):
    """
    Merge multiple overlapping patches into a single image with smooth blending.
    
    Parameters:
    -----------
    patches : list of numpy.ndarray
        List of patch images, each of size (patch_size, patch_size)
    patch_positions : list of tuples
        List of (y, x) positions for the top-left corner of each patch in the output image
    patch_size : int
        Size of each square patch (width/height)
    halo_size : int
        Size of the halo region
    output_shape : tuple
        Shape of the output image (height, width)
    grid_type : str, optional
        'cornered': Grid points at corners (traditional pixel-centered)
        'centered': Grid points at cell centers (grid cell-centered)
        
    Returns:
    --------
    numpy.ndarray
        Merged image
    """
    # Validate input
    assert all(p.shape[:2] == (patch_size, patch_size) for p in patches), "All patches must have the same size"
    
    # Determine if patches have a channel dimension
    if len(patches[0].shape) == 3:
        has_channels = True
        num_channels = patches[0].shape[2]
    else:
        has_channels = False
        num_channels = 1
    
    # Initialize output arrays
    output_height, output_width = output_shape
    if has_channels:
        merged_image = np.zeros((output_height, output_width, num_channels), dtype=np.float32)
        weights = np.zeros((output_height, output_width, num_channels), dtype=np.float32)
    else:
        merged_image = np.zeros((output_height, output_width), dtype=np.float32)
        weights = np.zeros_like(merged_image, dtype=np.float32)
    
    # Create weight mask based on grid type
    y, x = np.mgrid[0:patch_size, 0:patch_size]
    
    if grid_type == 'cornered':
        # Grid corners case - weight is highest at center and decreases towards edges
        center = patch_size // 2
        # Create a radial distance map from the center
        distance = np.sqrt((x - center)**2 + (y - center)**2)
        # Create Gaussian weight mask
        sigma = patch_size / 6  # Parameter controlling the falloff rate
        weight_mask = np.exp(-(distance**2) / (2 * sigma**2))
        
    elif grid_type == 'centered':
        # Grid centered case - core values have full weight, halo has decreasing weight
        # Create a mask where inner core has weight 1.0
        core_size = patch_size - 2 * halo_size
        weight_mask = np.ones((patch_size, patch_size), dtype=np.float32)
        
        # Apply linear falloff only in the halo regions
        for i in range(halo_size):
            # Top halo
            weight_mask[i, :] = (i + 1) / (halo_size + 1)
            # Bottom halo
            weight_mask[core_size + halo_size + i, :] = (halo_size - i) / (halo_size + 1)
            # Left halo
            weight_mask[:, i] = (i + 1) / (halo_size + 1)
            # Right halo
            weight_mask[:, core_size + halo_size + i] = (halo_size - i) / (halo_size + 1)
            
    else:
        raise ValueError("grid_type must be either 'cornered' or 'centered'")
    
    # Normalize weight mask to have maximum value of 1
    weight_mask = weight_mask / np.max(weight_mask)
    
    # For each patch
    for i, (patch, position) in enumerate(zip(patches, patch_positions)):
        y_start, x_start = position
        
        # Define the region in the output image where this patch will be placed
        y_end = min(y_start + patch_size, output_height)
        x_end = min(x_start + patch_size, output_width)
        
        # Handle patches that might extend beyond the output boundaries
        patch_y_end = y_end - y_start
        patch_x_end = x_end - x_start
        
        # Reshape weight mask if needed
        if has_channels:
            patch_weight = weight_mask[:patch_y_end, :patch_x_end, np.newaxis]
        else:
            patch_weight = weight_mask[:patch_y_end, :patch_x_end]
            if len(patch.shape) == 3:  # If patch has channels but we're processing as single channel
                patch = patch[:, :, 0]
        
        # Add weighted patch to the merged image
        merged_image[y_start:y_end, x_start:x_end] += patch[:patch_y_end, :patch_x_end] * patch_weight
        weights[y_start:y_end, x_start:x_end] += patch_weight
    
    # Normalize by the accumulated weights to get the final image
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    merged_image = merged_image / (weights + epsilon)
    
    return merged_image.astype(np.uint8) if patches[0].dtype == np.uint8 else merged_image


def generate_patch_positions(image_size, patch_size, core_size, grid_type='cornered'):
    """
    Generate patch positions for a regularly spaced grid.
    
    Parameters:
    -----------
    image_size : tuple
        Size of the full image (height, width)
    patch_size : int
        Size of each patch
    core_size : int
        Size of the core region (excluding halos)
    grid_type : str, optional
        'cornered': Grid points at corners (traditional pixel-centered)
        'centered': Grid points at cell centers (grid cell-centered)
        
    Returns:
    --------
    list of tuples
        List of (y, x) positions for each patch
    """
    height, width = image_size
    halo_size = (patch_size - core_size) // 2
    
    positions = []
    
    if grid_type == 'cornered':
        # For grid-cornered case: patches are positioned at regular intervals
        # with step size = core_size
        for y in range(0, height - core_size + 1, core_size):
            for x in range(0, width - core_size + 1, core_size):
                positions.append((y, x))
    
    elif grid_type == 'centered':
        # For grid-centered case: grid points are at centers of cells
        # First and last patches are positioned to include the image edges
        
        # Calculate centers for y
        y_centers = []
        y_pos = halo_size  # Start position
        while y_pos < height - halo_size:
            y_centers.append(y_pos)
            y_pos += core_size
        
        # Calculate centers for x
        x_centers = []
        x_pos = halo_size  # Start position
        while x_pos < width - halo_size:
            x_centers.append(x_pos)
            x_pos += core_size
        
        # Generate positions with patch centers at grid points
        for y_center in y_centers:
            for x_center in x_centers:
                # Convert center position to top-left corner position
                y_start = max(0, y_center - patch_size // 2)
                x_start = max(0, x_center - patch_size // 2)
                
                # Adjust if patch would extend beyond image boundary
                if y_start + patch_size > height:
                    y_start = height - patch_size
                if x_start + patch_size > width:
                    x_start = width - patch_size
                
                positions.append((y_start, x_start))
    
    else:
        raise ValueError("grid_type must be either 'cornered' or 'centered'")
    
    return positions


def extract_patches(image, patch_size, core_size, grid_type='cornered'):
    """
    Extract patches from a large image with specified overlap.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image to extract patches from
    patch_size : int
        Size of each patch
    core_size : int
        Size of the core region (excluding halos)
    grid_type : str, optional
        'cornered': Grid points at corners (traditional pixel-centered)
        'centered': Grid points at cell centers (grid cell-centered)
        
    Returns:
    --------
    tuple
        (patches, positions)
    """
    height, width = image.shape[:2]
    positions = generate_patch_positions((height, width), patch_size, core_size, grid_type)
    
    patches = []
    for y, x in positions:
        # Handle boundary conditions
        y_end = min(y + patch_size, height)
        x_end = min(x + patch_size, width)
        
        # Extract patch
        patch = image[y:y_end, x:x_end]
        
        # Pad if necessary (for patches at boundaries)
        if patch.shape[:2] != (patch_size, patch_size):
            if len(image.shape) == 3:
                padded_patch = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
            else:
                padded_patch = np.zeros((patch_size, patch_size), dtype=image.dtype)
            
            padded_patch[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded_patch
        
        patches.append(patch)
    
    return patches, positions


# Example usage
def demonstrate_grid_types():
    """
    Demonstrate the difference between grid-cornered and grid-centered approaches.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    # Create a sample image with gradient
    height, width = 600, 600
    y, x = np.mgrid[0:height, 0:width]
    image = np.sin(x/50) * np.cos(y/50)
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    
    # Parameters
    patch_size = 240
    core_size = 200
    halo_size = 20  # (patch_size - core_size) / 2
    
    # Extract and merge patches using both methods
    cornered_patches, cornered_positions = extract_patches(image, patch_size, core_size, 'cornered')
    centered_patches, centered_positions = extract_patches(image, patch_size, core_size, 'centered')
    
    merged_cornered = merge_patches(cornered_patches, cornered_positions, patch_size, halo_size, (height, width), 'cornered')
    merged_centered = merge_patches(centered_patches, centered_positions, patch_size, halo_size, (height, width), 'centered')
    
    # Calculate errors
    error_cornered = np.abs(image - merged_cornered)
    error_centered = np.abs(image - merged_centered)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(image, cmap='viridis')
    axes[0, 0].set_title('Original Image')
    
    # Grid-cornered
    axes[0, 1].imshow(merged_cornered, cmap='viridis')
    axes[0, 1].set_title('Grid-Cornered Merged')
    
    # Grid-centered
    axes[0, 2].imshow(merged_centered, cmap='viridis')
    axes[0, 2].set_title('Grid-Centered Merged')
    
    # Patch visualization
    axes[1, 0].imshow(image, cmap='viridis', alpha=0.7)
    axes[1, 0].set_title('Patch Positions')
    
    # Draw patches for grid-cornered
    for y, x in cornered_positions:
        rect = Rectangle((x, y), patch_size, patch_size, 
                         linewidth=1, edgecolor='r', facecolor='none', alpha=0.5)
        axes[1, 0].add_patch(rect)
        # Draw core regions
        rect_core = Rectangle((x + halo_size, y + halo_size), core_size, core_size, 
                             linewidth=1, edgecolor='g', facecolor='none', alpha=0.5)
        axes[1, 0].add_patch(rect_core)
    
    # Error visualization
    vmax = max(error_cornered.max(), error_centered.max())
    axes[1, 1].imshow(error_cornered, cmap='hot', vmin=0, vmax=vmax)
    axes[1, 1].set_title(f'Grid-Cornered Error (Max: {error_cornered.max():.6f})')
    
    axes[1, 2].imshow(error_centered, cmap='hot', vmin=0, vmax=vmax)
    axes[1, 2].set_title(f'Grid-Centered Error (Max: {error_centered.max():.6f})')
    
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    return {
        'cornered': {
            'patches': cornered_patches,
            'positions': cornered_positions,
            'merged': merged_cornered,
            'error': error_cornered
        },
        'centered': {
            'patches': centered_patches,
            'positions': centered_positions,
            'merged': merged_centered,
            'error': error_centered
        }
    }

import matplotlib.pyplot as plt
plt.plot(np.arange(10))
plt.show()
demonstrate_grid_types()

patch_size = 512
core_size = 472  # patch_size - 2*halo_size
halo_size = 20

your_image = np.random.random(512*512).reshape(512,512)
# Extract patches
patches, positions = extract_patches(
    image=your_image,
    patch_size=patch_size,
    core_size=core_size,
    grid_type='cornered'  # or 'centered'
)

# Merge patches
merged_image = merge_patches(
    patches=patches,
    patch_positions=positions,
    patch_size=patch_size,
    halo_size=halo_size,
    output_shape=your_image.shape[:2],
    grid_type='cornered'  # or 'centered'
)