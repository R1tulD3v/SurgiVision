from spleen_preprocessing import SpleenDataPreprocessor
import sys
import numpy as np

def test_data_loading():
    """Test if data loading works correctly"""
    try:
        # Initialize preprocessor
        data_root = "../data/Task09_Spleen"
        preprocessor = SpleenDataPreprocessor(data_root)
        
        print("=== Data Loading Test ===")
        print(f"Number of training images: {len(preprocessor.image_files)}")
        print(f"Number of training labels: {len(preprocessor.label_files)}")
        
        if len(preprocessor.image_files) == 0:
            print("❌ No image files found! Check your data path.")
            return False
            
        # Test loading first volume
        volume_path = preprocessor.image_files[0]
        mask_path = preprocessor.label_files[0]
        
        print(f"\nTesting with: {volume_path.name}")
        
        volume, mask = preprocessor.load_volume_and_mask(volume_path, mask_path)
        
        if volume is not None:
            print(f"✅ Successfully loaded volume: {volume.shape}")
            print(f"✅ Successfully loaded mask: {mask.shape}")
            print(f"Volume intensity range: {volume.min():.1f} to {volume.max():.1f}")
            print(f"Unique mask values: {np.unique(mask)}")
            return True
        else:
            print("❌ Failed to load volume")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_preprocessing():
    """Test preprocessing pipeline"""
    try:
        data_root = "../data/Task09_Spleen"
        preprocessor = SpleenDataPreprocessor(data_root)
        
        print("\n=== Preprocessing Test ===")
        
        # Test preprocessing
        volume_path = preprocessor.image_files[0]
        mask_path = preprocessor.label_files[0]
        
        processed_vol, processed_mask = preprocessor.preprocess_spleen_volume(
            volume_path, mask_path
        )
        
        if processed_vol is not None:
            print(f"✅ Preprocessing successful!")
            print(f"Processed volume shape: {processed_vol.shape}")
            print(f"Processed volume range: {processed_vol.min():.3f} to {processed_vol.max():.3f}")
            print(f"Non-zero spleen voxels: {np.sum(processed_mask > 0)}")
            return True
        else:
            print("❌ Preprocessing failed")
            return False
            
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Spleen Data Pipeline\n")
    
    # Test 1: Data loading
    loading_success = test_data_loading()
    
    if not loading_success:
        print("\n❌ Data loading failed. Please check:")
        print("1. Zone.Identifier files are removed")
        print("2. Data path is correct")
        print("3. NIfTI files are not corrupted")
        sys.exit(1)
    
    # Test 2: Preprocessing
    preprocessing_success = test_preprocessing()
    
    if preprocessing_success:
        print("\n🎉 All tests passed! Ready for 3D model development.")
    else:
        print("\n❌ Tests failed. Check the error messages above.")
