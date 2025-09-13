import nibabel as nib
import numpy as np
from scipy import ndimage
import os
from pathlib import Path
import matplotlib.pyplot as plt

class SpleenDataPreprocessor:
    def __init__(self, data_root):
        """
        Initialize spleen data preprocessor
        data_root: path to Task09_Spleen folder
        """
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "imagesTr"
        self.labels_dir = self.data_root / "labelsTr" 
        self.test_images_dir = self.data_root / "imagesTs"
        
        # Get file lists - FILTER OUT HIDDEN FILES
        all_image_files = list(self.images_dir.glob("*.nii.gz"))
        all_label_files = list(self.labels_dir.glob("*.nii.gz"))
        
        # Remove hidden files (starting with . or _)
        self.image_files = sorted([f for f in all_image_files 
                                  if not f.name.startswith('.') and not f.name.startswith('_')])
        self.label_files = sorted([f for f in all_label_files 
                                  if not f.name.startswith('.') and not f.name.startswith('_')])
        
        print(f"Found {len(self.image_files)} training images")
        print(f"Found {len(self.label_files)} training labels")
        
        # Verify matching pairs
        if len(self.image_files) > 0 and len(self.label_files) > 0:
            image_numbers = [f.name.split('_')[1].split('.')[0] for f in self.image_files]
            label_numbers = [f.name.split('_')[1].split('.')[0] for f in self.label_files] 
            
            if image_numbers != label_numbers:
                print("⚠️  Warning: Image and label files don't match!")
                print(f"Image IDs: {image_numbers[:5]}...")
                print(f"Label IDs: {label_numbers[:5]}...")
            else:
                print(f"✅ All {len(self.image_files)} image-label pairs matched")
        
    def load_volume_and_mask(self, volume_path, mask_path):
        """Load NIfTI volume and corresponding mask"""
        try:
            volume = nib.load(volume_path).get_fdata()
            mask = nib.load(mask_path).get_fdata()
            return volume, mask
        except Exception as e:
            print(f"Error loading {volume_path}: {e}")
            return None, None
    
    def preprocess_spleen_volume(self, volume_path, mask_path, target_size=(64, 64, 64)):
        """
        Preprocess spleen volume for 3D autoencoder
        """
        # Load NIfTI files
        volume, mask = self.load_volume_and_mask(volume_path, mask_path)
        if volume is None:
            return None, None
            
        print(f"Original volume shape: {volume.shape}")
        print(f"Original mask shape: {mask.shape}")
        
        # Find spleen bounding box
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            print("No spleen mask found!")
            return None, None
            
        # Add padding around spleen
        z_min = max(0, coords[2].min() - 5)
        z_max = min(volume.shape[2], coords[2].max() + 5)
        y_min = max(0, coords[1].min() - 10)
        y_max = min(volume.shape[1], coords[1].max() + 10)
        x_min = max(0, coords[0].min() - 10)
        x_max = min(volume.shape[0], coords[0].max() + 10)
        
        # Crop to spleen region
        cropped_volume = volume[x_min:x_max, y_min:y_max, z_min:z_max]
        cropped_mask = mask[x_min:x_max, y_min:y_max, z_min:z_max]
        
        print(f"Cropped volume shape: {cropped_volume.shape}")
        
        # Normalize CT intensities (Hounsfield Units)
        # Spleen tissue: approximately 40-60 HU
        volume_norm = np.clip(cropped_volume, -200, 300)  # Soft tissue window
        volume_norm = (volume_norm + 200) / 500  # Normalize to [0,1]
        
        # Resize to standard dimensions for GPU memory management
        zoom_factors = [target_size[i] / volume_norm.shape[i] for i in range(3)]
        volume_resized = ndimage.zoom(volume_norm, zoom_factors, order=1)
        mask_resized = ndimage.zoom(cropped_mask.astype(float), zoom_factors, order=0)
        
        print(f"Final volume shape: {volume_resized.shape}")
        
        return volume_resized, mask_resized
    
    def visualize_preprocessing(self, idx=0):
        """Visualize preprocessing results"""
        if idx >= len(self.image_files):
            print(f"Index {idx} out of range. Max: {len(self.image_files)-1}")
            return
            
        volume_path = self.image_files[idx]
        mask_path = self.label_files[idx]
        
        print(f"Processing: {volume_path.name}")
        
        # Original data
        original_volume, original_mask = self.load_volume_and_mask(volume_path, mask_path)
        
        # Preprocessed data  
        processed_volume, processed_mask = self.preprocess_spleen_volume(volume_path, mask_path)
        
        if processed_volume is None:
            return
            
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original data - middle slices
        mid_z_orig = original_volume.shape[2] // 2
        mid_z_proc = processed_volume.shape[2] // 2
        
        axes[0, 0].imshow(original_volume[:, :, mid_z_orig], cmap='gray')
        axes[0, 0].set_title(f'Original Volume\n{original_volume.shape}')
        
        axes[0, 1].imshow(original_mask[:, :, mid_z_orig], cmap='hot')
        axes[0, 1].set_title(f'Original Mask\n{original_mask.shape}')
        
        axes[0, 2].imshow(original_volume[:, :, mid_z_orig] * (original_mask[:, :, mid_z_orig] > 0), cmap='gray')
        axes[0, 2].set_title('Original Spleen Region')
        
        # Processed data
        axes[1, 0].imshow(processed_volume[:, :, mid_z_proc], cmap='gray')
        axes[1, 0].set_title(f'Processed Volume\n{processed_volume.shape}')
        
        axes[1, 1].imshow(processed_mask[:, :, mid_z_proc], cmap='hot')
        axes[1, 1].set_title(f'Processed Mask\n{processed_mask.shape}')
        
        axes[1, 2].imshow(processed_volume[:, :, mid_z_proc] * (processed_mask[:, :, mid_z_proc] > 0), cmap='gray')
        axes[1, 2].set_title('Processed Spleen Region')
        
        plt.tight_layout()
        plt.savefig(f'../results/preprocessing_example_{idx}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print(f"\nIntensity Statistics:")
        print(f"Original - Min: {original_volume.min():.1f}, Max: {original_volume.max():.1f}")
        print(f"Processed - Min: {processed_volume.min():.3f}, Max: {processed_volume.max():.3f}")
        print(f"Spleen voxels: {np.sum(processed_mask > 0)}")
    
    def prepare_training_data(self, output_dir="../data/preprocessed"):
        """Prepare and save preprocessed training data"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        processed_data = []
        
        for i, (img_path, mask_path) in enumerate(zip(self.image_files, self.label_files)):
            print(f"Processing {i+1}/{len(self.image_files)}: {img_path.name}")
            
            volume, mask = self.preprocess_spleen_volume(img_path, mask_path)
            
            if volume is not None:
                # Save preprocessed data
                np.save(output_path / f"volume_{i:03d}.npy", volume)
                np.save(output_path / f"mask_{i:03d}.npy", mask)
                
                processed_data.append({
                    'volume_path': str(output_path / f"volume_{i:03d}.npy"),
                    'mask_path': str(output_path / f"mask_{i:03d}.npy"),
                    'original_id': img_path.stem,
                })
        
        # Save metadata
        import json
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(processed_data, f, indent=2)
            
        print(f"\nProcessed {len(processed_data)} volumes")
        print(f"Saved to: {output_path}")
        
        return processed_data

# Usage example and testing
if __name__ == "__main__":
    # Initialize preprocessor
    data_root = "../data/Task09_Spleen"  # Relative to src/ folder
    preprocessor = SpleenDataPreprocessor(data_root)
    
    # Test preprocessing on first sample
    print("=== Testing Preprocessing ===")
    preprocessor.visualize_preprocessing(idx=0)
    
    # Prepare training data (uncomment when ready)
    # print("\n=== Preparing All Training Data ===")
    # training_data = preprocessor.prepare_training_data()
