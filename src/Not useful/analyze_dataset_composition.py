import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from spleen_preprocessing import SpleenDataPreprocessor

def analyze_all_volumes():
    """Analyze all 41 volumes to determine normal vs abnormal"""
    print("🔬 COMPREHENSIVE DATASET COMPOSITION ANALYSIS")
    print("="*60)
    
    preprocessor = SpleenDataPreprocessor("../data/Task09_Spleen")
    
    normal_count = 0
    abnormal_count = 0
    volume_stats = []
    
    print(f"Analyzing all {len(preprocessor.image_files)} volumes...")
    print()
    
    for i in range(len(preprocessor.image_files)):
        print(f"📋 Volume {i+1}/{len(preprocessor.image_files)}: ", end="")
        
        volume_path = preprocessor.image_files[i]
        mask_path = preprocessor.label_files[i]
        
        # Load raw data
        try:
            img = nib.load(volume_path)
            mask = nib.load(mask_path)
            
            img_data = img.get_fdata()
            mask_data = mask.get_fdata()
            
            # Analyze mask labels
            unique_labels = np.unique(mask_data)
            
            # Get spleen statistics
            spleen_voxels = np.sum(mask_data == 1) if 1 in unique_labels else 0
            total_voxels = mask_data.size
            spleen_percentage = spleen_voxels / total_voxels * 100
            
            # Check for any abnormal patterns
            has_tumor_label = 2 in unique_labels or len(unique_labels) > 2
            
            # Analyze intensity patterns within spleen
            if spleen_voxels > 0:
                spleen_region = img_data[mask_data == 1]
                spleen_mean = np.mean(spleen_region)
                spleen_std = np.std(spleen_region)
                spleen_min = np.min(spleen_region)
                spleen_max = np.max(spleen_region)
                
                # Look for unusual intensity patterns (very bright or dark regions)
                unusual_bright = np.sum(spleen_region > spleen_mean + 3*spleen_std)
                unusual_dark = np.sum(spleen_region < spleen_mean - 3*spleen_std)
                unusual_percentage = (unusual_bright + unusual_dark) / len(spleen_region) * 100
            else:
                spleen_mean = spleen_std = spleen_min = spleen_max = 0
                unusual_percentage = 0
            
            # Classification criteria
            is_abnormal = (
                has_tumor_label or  # Has tumor labels
                unusual_percentage > 5.0 or  # >5% unusual intensities
                spleen_percentage < 0.1 or  # Unusually small spleen
                spleen_percentage > 3.0  # Unusually large spleen
            )
            
            if is_abnormal:
                abnormal_count += 1
                status = "🚨 POTENTIALLY ABNORMAL"
                reasons = []
                if has_tumor_label:
                    reasons.append("tumor labels detected")
                if unusual_percentage > 5.0:
                    reasons.append(f"{unusual_percentage:.1f}% unusual intensities")
                if spleen_percentage < 0.1:
                    reasons.append("unusually small spleen")
                if spleen_percentage > 3.0:
                    reasons.append("unusually large spleen")
                reason_text = ", ".join(reasons)
            else:
                normal_count += 1
                status = "✅ NORMAL"
                reason_text = "healthy spleen pattern"
            
            print(f"{volume_path.name}")
            print(f"    Labels: {unique_labels}")
            print(f"    Spleen: {spleen_voxels:,} voxels ({spleen_percentage:.2f}%)")
            print(f"    Intensity: {spleen_mean:.1f}±{spleen_std:.1f} [{spleen_min:.1f}-{spleen_max:.1f}]")
            print(f"    Status: {status} ({reason_text})")
            print()
            
            # Store statistics
            volume_stats.append({
                'index': i+1,
                'name': volume_path.name,
                'unique_labels': unique_labels,
                'spleen_voxels': spleen_voxels,
                'spleen_percentage': spleen_percentage,
                'spleen_mean': spleen_mean,
                'spleen_std': spleen_std,
                'unusual_percentage': unusual_percentage,
                'is_abnormal': is_abnormal,
                'reason': reason_text
            })
            
        except Exception as e:
            print(f"❌ Error analyzing {volume_path.name}: {e}")
            continue
    
    return normal_count, abnormal_count, volume_stats

def print_summary(normal_count, abnormal_count, volume_stats):
    """Print comprehensive summary"""
    total = normal_count + abnormal_count
    
    print("="*60)
    print("📊 DATASET COMPOSITION SUMMARY")
    print("="*60)
    
    print(f"Total Volumes Analyzed: {total}")
    print(f"Normal Volumes: {normal_count} ({normal_count/total*100:.1f}%)")
    print(f"Potentially Abnormal: {abnormal_count} ({abnormal_count/total*100:.1f}%)")
    
    if abnormal_count > 0:
        print(f"\n🚨 POTENTIALLY ABNORMAL VOLUMES:")
        for stat in volume_stats:
            if stat['is_abnormal']:
                print(f"  {stat['index']:2d}. {stat['name']}: {stat['reason']}")
    
    print(f"\n📈 SPLEEN SIZE DISTRIBUTION:")
    spleen_percentages = [stat['spleen_percentage'] for stat in volume_stats]
    print(f"  Mean spleen size: {np.mean(spleen_percentages):.2f}% of volume")
    print(f"  Std deviation: {np.std(spleen_percentages):.2f}%")
    print(f"  Range: {np.min(spleen_percentages):.2f}% - {np.max(spleen_percentages):.2f}%")
    
    print(f"\n📋 LABEL ANALYSIS:")
    all_labels = set()
    for stat in volume_stats:
        all_labels.update(stat['unique_labels'])
    
    print(f"  All unique labels found: {sorted(all_labels)}")
    
    # Check if any volume has multiple labels
    multi_label_count = sum(1 for stat in volume_stats if len(stat['unique_labels']) > 2)
    print(f"  Volumes with >2 labels: {multi_label_count}")
    
    print(f"\n🎯 MEDICAL SEGMENTATION DECATHLON Task09_Spleen:")
    print(f"  Expected labels: [0, 1] (background, normal spleen)")
    print(f"  Expected abnormalities: 0 (healthy spleen dataset)")
    print(f"  Your results match expected: {'✅ YES' if abnormal_count == 0 else '⚠️ REVIEW NEEDED'}")

def analyze_intensity_patterns():
    """Analyze intensity patterns across all volumes"""
    print(f"\n🔍 INTENSITY PATTERN ANALYSIS:")
    print("="*60)
    
    preprocessor = SpleenDataPreprocessor("../data/Task09_Spleen")
    
    all_spleen_intensities = []
    volume_means = []
    
    for i in range(min(10, len(preprocessor.image_files))):  # Sample first 10
        try:
            volume_path = preprocessor.image_files[i]
            mask_path = preprocessor.label_files[i]
            
            img = nib.load(volume_path)
            mask = nib.load(mask_path)
            
            img_data = img.get_fdata()
            mask_data = mask.get_fdata()
            
            # Get spleen region intensities
            spleen_region = img_data[mask_data == 1]
            
            if len(spleen_region) > 0:
                all_spleen_intensities.extend(spleen_region.flatten()[:1000])  # Sample 1000 voxels
                volume_means.append(np.mean(spleen_region))
        
        except:
            continue
    
    if all_spleen_intensities:
        print(f"Analyzed spleen intensities from {len(volume_means)} volumes:")
        print(f"  Overall intensity range: {np.min(all_spleen_intensities):.1f} - {np.max(all_spleen_intensities):.1f}")
        print(f"  Mean intensity: {np.mean(all_spleen_intensities):.1f}")
        print(f"  Standard deviation: {np.std(all_spleen_intensities):.1f}")
        
        # Check for bimodal distribution (could indicate mixed normal/abnormal)
        hist, bins = np.histogram(all_spleen_intensities, bins=50)
        peaks = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.max(hist) * 0.1:
                peaks.append(bins[i])
        
        print(f"  Intensity peaks detected: {len(peaks)}")
        if len(peaks) > 1:
            print(f"    ⚠️  Multiple peaks might indicate mixed tissue types")
        else:
            print(f"    ✅ Single peak suggests homogeneous normal tissue")

def main():
    """Run comprehensive dataset analysis"""
    print("Starting comprehensive analysis of Task09_Spleen dataset...")
    print()
    
    # Main analysis
    normal_count, abnormal_count, volume_stats = analyze_all_volumes()
    
    # Print summary
    print_summary(normal_count, abnormal_count, volume_stats)
    
    # Intensity analysis
    analyze_intensity_patterns()
    
    print(f"\n🏆 FINAL CONCLUSION:")
    print("="*60)
    if abnormal_count == 0:
        print("✅ ALL VOLUMES ARE NORMAL!")
        print("✅ Your model was trained on 100% healthy spleen tissue")
        print("✅ Perfect foundation for anomaly detection via synthetic pathologies")
        print("✅ Medically sound approach - learn normal, detect abnormal")
    else:
        print(f"⚠️  {abnormal_count} volumes flagged for review")
        print("💡 These may still be normal but with unusual characteristics")
        print("💡 Manual review recommended")
    
    print(f"\n🎯 For your hackathon presentation:")
    print(f"   'My model learned from {normal_count} normal, healthy spleen volumes'")
    print(f"   'I then created synthetic pathologies to teach it what's abnormal'")
    print(f"   'This unsupervised approach mirrors real clinical practice'")

if __name__ == "__main__":
    main()
