import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Spleen3DAutoencoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=512):
        super(Spleen3DAutoencoder, self).__init__()
        
        # 3D Encoder - optimized for 4GB GPU
        self.encoder = nn.Sequential(
            # Input: 1×64×64×64 → 16×32×32×32
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # 16×32×32×32 → 32×16×16×16  
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # 32×16×16×16 → 64×8×8×8
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        
        # Bottleneck - compressed representation
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8 * 8, latent_dim),  # 32768 → 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Prevent overfitting
            nn.Linear(latent_dim, 64 * 8 * 8 * 8),  # 512 → 32768
            nn.ReLU(inplace=True),
        )
        
        # 3D Decoder - reconstruct volume
        self.decoder = nn.Sequential(
            # 64×8×8×8 → 32×16×16×16
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # 32×16×16×16 → 16×32×32×32
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            
            # 16×32×32×32 → 1×64×64×64
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output range [0,1]
        )
        
        self.latent_dim = latent_dim
        
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        batch_size = encoded.size(0)
        
        # Bottleneck  
        flattened = encoded.view(batch_size, -1)
        bottleneck_out = self.bottleneck(flattened)
        reshaped = bottleneck_out.view(batch_size, 64, 8, 8, 8)
        
        # Decode
        decoded = self.decoder(reshaped)
        return decoded
    
    def encode(self, x):
        """Get encoded features for anomaly detection"""
        encoded = self.encoder(x)
        flattened = encoded.view(encoded.size(0), -1)
        # Get bottleneck features before reshaping
        features = self.bottleneck[0](flattened)  # Linear layer only
        return features

def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def test_model_architecture():
    """Test model architecture and memory usage"""
    print("=== 3D Spleen Autoencoder Architecture Test ===")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Spleen3DAutoencoder().to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Test forward pass
    batch_size = 2 if device.type == 'cuda' else 1
    dummy_input = torch.randn(batch_size, 1, 64, 64, 64).to(device)
    
    print(f"\nTesting with batch size {batch_size} on {device}")
    print(f"Input shape: {dummy_input.shape}")
    
    try:
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
            encoded_features = model.encode(dummy_input)
        
        print(f"✅ Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Encoded features shape: {encoded_features.shape}")
        
        # Memory usage (if CUDA)
        if device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated(device) / 1024**3
            memory_cached = torch.cuda.memory_reserved(device) / 1024**3
            print(f"GPU memory used: {memory_used:.2f} GB")
            print(f"GPU memory cached: {memory_cached:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

if __name__ == "__main__":
    test_model_architecture()
