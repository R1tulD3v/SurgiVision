import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time
from spleen_preprocessing import SpleenDataPreprocessor
from spleen_3d_model import Spleen3DAutoencoder

class SpleenDataset(Dataset):
    def __init__(self, preprocessor, train=True, normal_only=True):
        self.preprocessor = preprocessor
        self.normal_only = normal_only
        
        # Use 80% for training, 20% for validation
        n_files = len(preprocessor.image_files)
        split_idx = int(0.8 * n_files)
        
        if train:
            self.image_files = preprocessor.image_files[:split_idx]
            self.label_files = preprocessor.label_files[:split_idx]
        else:
            self.image_files = preprocessor.image_files[split_idx:]
            self.label_files = preprocessor.label_files[split_idx:]
        
        print(f"{'Training' if train else 'Validation'} dataset: {len(self.image_files)} volumes")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        volume_path = self.image_files[idx]
        mask_path = self.label_files[idx]
        
        # Preprocess volume
        volume, mask = self.preprocessor.preprocess_spleen_volume(volume_path, mask_path)
        
        if volume is None:
            # Return dummy data if preprocessing fails
            volume = np.zeros((64, 64, 64))
            mask = np.zeros((64, 64, 64))
        
        # Convert to torch tensors
        volume_tensor = torch.FloatTensor(volume[np.newaxis, ...])  # Add channel dim
        
        if self.normal_only:
            # For autoencoder: only return spleen regions (normal tissue)
            spleen_mask = torch.FloatTensor(mask) > 0
            masked_volume = volume_tensor.clone()
            # Zero out non-spleen regions
            masked_volume[0, ~spleen_mask] = 0
            return masked_volume
        else:
            # Return full volume and mask
            mask_tensor = torch.FloatTensor(mask[np.newaxis, ...])
            return volume_tensor, mask_tensor

class SpleenTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training configuration
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, volumes in enumerate(self.train_loader):
            volumes = volumes.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed = self.model(volumes)
            loss = self.criterion(reconstructed, volumes)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Memory cleanup for GPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx+1}: Loss = {loss.item():.6f}")
        
        return epoch_loss / num_batches
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for volumes in self.val_loader:
                volumes = volumes.to(self.device)
                reconstructed = self.model(volumes)
                loss = self.criterion(reconstructed, volumes)
                val_loss += loss.item()
                num_batches += 1
        
        return val_loss / num_batches
    
    def train(self, num_epochs=30):
        """Full training loop"""
        print(f"Starting training for {num_epochs} epochs on {self.device}")
        start_time = time.time()
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation  
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, '../models/best_spleen_3d_autoencoder.pth')
                print(f"✅ New best model saved (val_loss: {val_loss:.6f})")
            
            # Early stopping
            if current_lr < 1e-6:
                print("Learning rate too small, stopping training")
                break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.1f} seconds")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        return self.train_losses, self.val_losses

def main():
    """Main training function"""
    print("=== 3D Spleen Autoencoder Training ===")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize preprocessor
    data_root = "../data/Task09_Spleen"
    preprocessor = SpleenDataPreprocessor(data_root)
    
    # Create datasets
    train_dataset = SpleenDataset(preprocessor, train=True, normal_only=True)
    val_dataset = SpleenDataset(preprocessor, train=False, normal_only=True)
    
    # Create data loaders
    batch_size = 2 if device.type == 'cuda' else 1  # Adjust for GPU memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create model
    model = Spleen3DAutoencoder()
    
    # Create trainer
    trainer = SpleenTrainer(model, train_loader, val_loader, device)
    
    # Train model
    train_losses, val_losses = trainer.train(num_epochs=25)
    
    print("\n✅ Training completed successfully!")
    print("Model saved to: ../models/best_spleen_3d_autoencoder.pth")

if __name__ == "__main__":
    main()
