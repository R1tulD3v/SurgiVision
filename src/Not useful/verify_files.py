from pathlib import Path
import nibabel as nib

def verify_spleen_files():
    """Verify all spleen files are valid"""
    data_root = Path("../data/Task09_Spleen")
    images_dir = data_root / "imagesTr"
    labels_dir = data_root / "labelsTr"
    
    # Get all .nii.gz files, excluding hidden files
    image_files = [f for f in images_dir.glob("*.nii.gz") 
                   if not f.name.startswith('.') and not f.name.startswith('_')]
    label_files = [f for f in labels_dir.glob("*.nii.gz")
                   if not f.name.startswith('.') and not f.name.startswith('_')]
    
    print(f"=== File Verification ===")
    print(f"Valid image files: {len(image_files)}")
    print(f"Valid label files: {len(label_files)}")
    
    # Sort and check pairing
    image_files = sorted(image_files)
    label_files = sorted(label_files)
    
    print(f"\nFirst 5 image files:")
    for f in image_files[:5]:
        print(f"  {f.name}")
        
    print(f"\nFirst 5 label files:")
    for f in label_files[:5]:
        print(f"  {f.name}")
    
    # Test loading a few files
    print(f"\n=== Testing File Loading ===")
    valid_count = 0
    
    for i, (img_file, lbl_file) in enumerate(zip(image_files[:3], label_files[:3])):
        try:
            # Test loading
            img_data = nib.load(img_file).get_fdata()
            lbl_data = nib.load(lbl_file).get_fdata()
            
            print(f"✅ {img_file.name}: {img_data.shape}, range: {img_data.min():.0f} to {img_data.max():.0f}")
            print(f"✅ {lbl_file.name}: {lbl_data.shape}, unique values: {len(set(lbl_data.flat))}")
            valid_count += 1
            
        except Exception as e:
            print(f"❌ {img_file.name}: Error - {e}")
    
    if valid_count == 3:
        print(f"\n🎉 All test files loaded successfully!")
        return True
    else:
        print(f"\n❌ Some files failed to load")
        return False

if __name__ == "__main__":
    verify_spleen_files()
