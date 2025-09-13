import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from spleen_3d_model import Spleen3DAutoencoder
from spleen_preprocessing import SpleenDataPreprocessor
from enhanced_anomaly_creator import MedicalAnomalyCreator

class HybridAnomalyDetector(nn.Module):
    def __init__(self, autoencoder_path):
        super(HybridAnomalyDetector, self).__init__()
        
        # Load pre-trained autoencoder
        self.autoencoder = Spleen3DAutoencoder()
        checkpoint = torch.load(autoencoder_path, map_location='cpu')
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze autoencoder weights
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        
        # Get the correct encoded feature size by testing
        test_input = torch.randn(1, 1, 64, 64, 64)
        with torch.no_grad():
            encoded_features = self.get_encoded_features(test_input)
            encoded_size = encoded_features.shape[1]
        
        print(f"Detected encoded feature size: {encoded_size}")
        
        # Classification head on encoded features (using correct size)
        self.anomaly_classifier = nn.Sequential(
            nn.Linear(encoded_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Binary: normal vs anomaly
        )
        
        # Spatial error pattern analyzer (3D CNN on reconstruction error)
        self.spatial_analyzer = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),  # 32x32x32
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),  # 16x16x16
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(4),  # 4x4x4
            
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        
        # Attention mechanism for error localization
        self.attention = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        print("✅ Hybrid model architecture created with correct dimensions")
    
    def get_encoded_features(self, x):
        """Safely extract encoded features from autoencoder"""
        # Use the encoder part of the autoencoder
        encoded = self.autoencoder.encoder(x)
        batch_size = encoded.size(0)
        
        # Flatten the encoded features
        flattened = encoded.view(batch_size, -1)
        
        # Apply the bottleneck to get compressed features
        bottleneck_out = self.autoencoder.bottleneck(flattened)
        
        # Extract the compressed features (first linear layer output)
        compressed_features = bottleneck_out
        
        return compressed_features
    
    def forward(self, x):
        # Get autoencoder reconstruction
        with torch.no_grad():
            reconstructed = self.autoencoder(x)
            encoded_features = self.get_encoded_features(x)
        
        # Method 1: Classification from encoded features
        classification_logits = self.anomaly_classifier(encoded_features)
        classification_probs = torch.softmax(classification_logits, dim=1)
        
        # Method 2: Spatial error analysis
        error_map = torch.abs(x - reconstructed)
        spatial_score = self.spatial_analyzer(error_map)
        
        # Method 3: Attention-weighted error analysis  
        attention_weights = self.attention(error_map)
        weighted_error = torch.mean(error_map * attention_weights, dim=[2,3,4])
        
        # Method 4: Global reconstruction error (original method)
        global_error = torch.mean((x - reconstructed) ** 2, dim=[1,2,3,4])
        
        return {
            'reconstructed': reconstructed,
            'classification_logits': classification_logits,
            'classification_probs': classification_probs,
            'spatial_score': spatial_score,
            'weighted_error': weighted_error,
            'global_error': global_error,
            'error_map': error_map,
            'attention_map': attention_weights
        }

class HybridDataset(Dataset):
    def __init__(self, preprocessor, split='train'):
        self.preprocessor = preprocessor
        
        # Create train/val split
        n_files = len(preprocessor.image_files)
        split_idx = int(0.8 * n_files)
        
        if split == 'train':
            self.indices = list(range(0, split_idx))
        else:
            self.indices = list(range(split_idx, n_files))
        
        # Create anomaly creator for this split
        self.anomaly_creator = MedicalAnomalyCreator(preprocessor)
        
        print(f"HybridDataset {split}: {len(self.indices)} base volumes")
    
    def __len__(self):
        return len(self.indices) * 2  # Each volume creates normal + anomalous pair
    
    def __getitem__(self, idx):
        base_idx = idx // 2
        is_anomalous = idx % 2 == 1
        
        file_idx = self.indices[base_idx]
        
        # Avoid index out of range
        if file_idx >= len(self.preprocessor.image_files):
            file_idx = self.indices[0]  # Fall back to first file
        
        volume_path = self.preprocessor.image_files[file_idx]
        mask_path = self.preprocessor.label_files[file_idx]
        
        # Preprocess base volume
        volume, mask = self.preprocessor.preprocess_spleen_volume(volume_path, mask_path)
        
        if volume is None:
            volume = np.zeros((64, 64, 64))
            mask = np.zeros((64, 64, 64))
        
        if is_anomalous:
            # Create synthetic pathology
            pathologies = self.anomaly_creator.create_all_pathologies(file_idx)
            if pathologies:
                # Randomly select one pathology type
                np.random.seed(idx)  # Reproducible random selection
                pathology_idx = np.random.randint(len(pathologies))
                volume = pathologies[pathology_idx]['volume']
            label = 1  # Anomalous
        else:
            label = 0  # Normal
        
        # Apply spleen mask (only analyze spleen region)
        spleen_mask = mask > 0
        masked_volume = volume.copy()
        masked_volume[~spleen_mask] = 0
        
        volume_tensor = torch.FloatTensor(masked_volume[np.newaxis, ...])  # Add channel dim
        
        return volume_tensor, label

class HybridTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Only train the new components (autoencoder frozen)
        trainable_params = []
        for name, param in model.named_parameters():
            if 'autoencoder' not in name and param.requires_grad:
                trainable_params.append(param)
        
        print(f"Training {len(trainable_params)} parameter groups (autoencoder frozen)")
        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.MSELoss()
        
        # Optimizer for trainable components only
        self.optimizer = optim.Adam(trainable_params, lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=3
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        self.model.train()
        
        # Keep autoencoder in eval mode
        self.model.autoencoder.eval()
        
        total_loss = 0
        classification_loss_sum = 0
        spatial_loss_sum = 0
        num_batches = 0
        
        for batch_idx, (volumes, labels) in enumerate(self.train_loader):
            volumes = volumes.to(self.device)
            labels = torch.LongTensor(labels).to(self.device)
            
            self.optimizer.zero_grad()
            
            try:
                # Forward pass
                outputs = self.model(volumes)
                
                # Multi-task loss
                classification_loss = self.classification_criterion(
                    outputs['classification_logits'], labels
                )
                
                # Spatial score should match binary labels
                spatial_targets = labels.float().unsqueeze(1)
                spatial_loss = self.regression_criterion(
                    torch.sigmoid(outputs['spatial_score']), spatial_targets
                )
                
                # Combined loss
                total_loss_batch = classification_loss + 0.5 * spatial_loss
                
                # Backward pass
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += total_loss_batch.item()
                classification_loss_sum += classification_loss.item()
                spatial_loss_sum += spatial_loss.item()
                num_batches += 1
                
                if batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx+1}: Loss = {total_loss_batch.item():.4f}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
            
            # Memory cleanup
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_class_loss = classification_loss_sum / num_batches
            avg_spatial_loss = spatial_loss_sum / num_batches
        else:
            avg_loss = avg_class_loss = avg_spatial_loss = 0
        
        return avg_loss, avg_class_loss, avg_spatial_loss
    
    def validate(self):
        self.model.eval()
        
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        num_batches = 0
        
        with torch.no_grad():
            for volumes, labels in self.val_loader:
                volumes = volumes.to(self.device)
                labels = torch.LongTensor(labels).to(self.device)
                
                try:
                    outputs = self.model(volumes)
                    
                    # Classification loss
                    classification_loss = self.classification_criterion(
                        outputs['classification_logits'], labels
                    )
                    
                    # Spatial loss
                    spatial_targets = labels.float().unsqueeze(1)
                    spatial_loss = self.regression_criterion(
                        torch.sigmoid(outputs['spatial_score']), spatial_targets
                    )
                    
                    total_loss_batch = classification_loss + 0.5 * spatial_loss
                    total_loss += total_loss_batch.item()
                    
                    # Accuracy calculation
                    predictions = torch.argmax(outputs['classification_probs'], dim=1)
                    correct_predictions += (predictions == labels).sum().item()
                    total_predictions += labels.size(0)
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Validation error: {e}")
                    continue
        
        if num_batches > 0 and total_predictions > 0:
            avg_loss = total_loss / num_batches
            accuracy = correct_predictions / total_predictions * 100
        else:
            avg_loss = 0
            accuracy = 0
        
        return avg_loss, accuracy
    
    def train(self, num_epochs=12):
        print(f"🚀 Training hybrid components for {num_epochs} epochs")
        
        best_val_accuracy = 0
        
        for epoch in range(num_epochs):
            print(f"\n📚 Epoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss, class_loss, spatial_loss = self.train_epoch()
            
            # Validation
            val_loss, val_accuracy = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Train Loss: {train_loss:.4f} (Class: {class_loss:.4f}, Spatial: {spatial_loss:.4f})")
            print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.1f}% | LR: {current_lr:.2e}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'accuracy': val_accuracy,
                    'loss': val_loss,
                }, '../models/best_hybrid_detector.pth')
                print(f"✅ New best hybrid model saved! Accuracy: {val_accuracy:.1f}%")
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping
            if current_lr < 1e-6:
                print("Learning rate too small, stopping training")
                break
        
        print(f"\n🎯 Training completed! Best validation accuracy: {best_val_accuracy:.1f}%")
        return best_val_accuracy

class AdvancedAnomalyPipeline:
    def __init__(self, hybrid_model_path=None, global_threshold=0.008756):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_threshold = global_threshold
        
        # Load hybrid model
        self.hybrid_model = HybridAnomalyDetector("../models/best_spleen_3d_autoencoder.pth")
        
        if hybrid_model_path and Path(hybrid_model_path).exists():
            checkpoint = torch.load(hybrid_model_path, map_location=self.device)
            self.hybrid_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded trained hybrid model (accuracy: {checkpoint.get('accuracy', 'unknown'):.1f}%)")
        else:
            print(f"⚠️  Using autoencoder-only mode (hybrid model not trained yet)")
        
        self.hybrid_model.to(self.device)
        self.hybrid_model.eval()
    
    def detect_with_ensemble(self, volume_tensor):
        """Advanced multi-modal anomaly detection"""
        volume_tensor = volume_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.hybrid_model(volume_tensor)
            
            # Method 1: Global reconstruction error (proven method)
            global_error = outputs['global_error'].item()
            method1_score = global_error / self.global_threshold
            method1_anomaly = global_error > self.global_threshold
            
            # Method 2: Deep learning classification
            class_probs = outputs['classification_probs'][0]
            method2_score = class_probs[1].item()  # Anomaly probability
            method2_anomaly = method2_score > 0.5
            
            # Method 3: Spatial pattern analysis
            spatial_score = torch.sigmoid(outputs['spatial_score']).item()
            method3_anomaly = spatial_score > 0.5
            
            # Method 4: Attention-weighted error analysis
            weighted_error = outputs['weighted_error'].item()
            method4_score = weighted_error / self.global_threshold
            method4_anomaly = weighted_error > self.global_threshold
            
            # Ensemble decision
            ensemble_score = (
                method1_score * 0.40 +    # Global error (proven)
                method2_score * 0.25 +    # Classification
                spatial_score * 0.20 +    # Spatial patterns
                method4_score * 0.15      # Attention error
            )
            
            final_anomaly = ensemble_score > 1.0
            
            return {
                'final_decision': final_anomaly,
                'ensemble_score': ensemble_score,
                'confidence_level': "HIGH" if ensemble_score > 1.5 else ("MEDIUM" if ensemble_score > 1.0 else "LOW"),
                'method_breakdown': {
                    'global_reconstruction': {'anomaly': method1_anomaly, 'score': method1_score},
                    'deep_classification': {'anomaly': method2_anomaly, 'score': method2_score},
                    'spatial_patterns': {'anomaly': method3_anomaly, 'score': spatial_score},
                    'weighted_attention': {'anomaly': method4_anomaly, 'score': method4_score}
                }
            }

def train_hybrid_system():
    """Train the hybrid anomaly detection system"""
    print("🔬 TRAINING ADVANCED HYBRID ANOMALY DETECTOR")
    print("="*60)
    
    # Check base model
    if not Path("../models/best_spleen_3d_autoencoder.pth").exists():
        print("❌ Base autoencoder not found!")
        return 0
    
    # Initialize components
    preprocessor = SpleenDataPreprocessor("../data/Task09_Spleen")
    
    # Create datasets
    train_dataset = HybridDataset(preprocessor, 'train')
    val_dataset = HybridDataset(preprocessor, 'val')
    
    # Data loaders (smaller batch size to avoid memory issues)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Initialize hybrid model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hybrid_model = HybridAnomalyDetector("../models/best_spleen_3d_autoencoder.pth")
    
    # Train hybrid components
    trainer = HybridTrainer(hybrid_model, train_loader, val_loader, device)
    best_accuracy = trainer.train(num_epochs=12)
    
    print("✅ Hybrid training completed!")
    return best_accuracy

def test_hybrid_system():
    """Test the hybrid system"""
    print("\n🧪 TESTING HYBRID ANOMALY DETECTION SYSTEM")
    print("="*60)
    
    # Test with enhanced anomalies
    pipeline = AdvancedAnomalyPipeline("../models/best_hybrid_detector.pth")
    
    print("🩺 Testing hybrid system on pathological cases...")
    
    # Create test case
    preprocessor = SpleenDataPreprocessor("../data/Task09_Spleen")
    anomaly_creator = MedicalAnomalyCreator(preprocessor)
    
    pathological_cases = anomaly_creator.create_all_pathologies(base_index=5)
    
    correct_detections = 0
    for i, case in enumerate(pathological_cases[:3]):  # Test first 3 cases
        print(f"\nCase {i+1}: {case['description']}")
        
        spleen_mask = case['mask'] > 0
        masked_volume = case['volume'].copy()
        masked_volume[~spleen_mask] = 0
        volume_tensor = torch.FloatTensor(masked_volume[np.newaxis, np.newaxis, ...])
        
        result = pipeline.detect_with_ensemble(volume_tensor)
        
        if result['final_decision']:
            correct_detections += 1
            status = "✅ DETECTED"
        else:
            status = "❌ MISSED"
        
        print(f"   Result: {status}")
        print(f"   Ensemble Score: {result['ensemble_score']:.3f}")
        print(f"   Confidence: {result['confidence_level']}")
    
    detection_rate = correct_detections / 3 * 100
    print(f"\n🎯 Detection Rate: {correct_detections}/3 ({detection_rate:.1f}%)")
    
    return detection_rate

def main():
    """Main hybrid system pipeline"""
    print("🚀 ADVANCED HYBRID ANOMALY DETECTION SYSTEM")
    print("="*60)
    
    try:
        # Phase 1: Train hybrid components
        print("Phase 1: Training hybrid components...")
        best_accuracy = train_hybrid_system()
        
        if best_accuracy > 0:
            print(f"✅ Training successful! Best accuracy: {best_accuracy:.1f}%")
            
            # Phase 2: Test system
            print("\nPhase 2: Testing hybrid system...")
            detection_rate = test_hybrid_system()
            
            print(f"\n🎉 Hybrid system ready!")
            print(f"💡 Training accuracy: {best_accuracy:.1f}%")
            print(f"💡 Test detection rate: {detection_rate:.1f}%")
            
        else:
            print("❌ Training failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
