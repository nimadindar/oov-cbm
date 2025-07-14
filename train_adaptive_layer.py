import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import clip
import numpy as np
from tqdm import tqdm
import os

# --- Load CB-AE ---
from sparse_autoencoder import SparseAutoencoder


# --- Load your dataset ---
from elvish_load.load_elvish_dataset import load_elvish_dataset

class TextEmbeddingDataset(Dataset):
    def __init__(self, text_pairs, clip_model, device):
        self.text_pairs = text_pairs  # List of (t, t_eng) pairs
        self.clip_model = clip_model
        self.device = device
        self.cb = SparseAutoencoder(
                n_input_features=512,
                n_learned_features=4096,
                n_components=1)
        
        self.cb.load_state_dict(torch.load("./DNCBM/Checkpoints/clip_ViT-B_16_sparse_autoencoder_final.pt"))
        self.cb.to(self.device)
        self.cb.eval()
        self.preprocess_text()  # Precompute CLIP embeddings
        

    def preprocess_text(self):
        # TODO Nima: This currently encodes text into CLIP embedding. 
        # We need to encode the CLIP embedding further into concept bottleneck layers.
        # This can be done simply by calling the cbae on CLIP embedding.
        self.embeddings_t = []
        self.embeddings_t_eng = []
        for t, t_eng in tqdm(self.text_pairs, desc="Encoding text with CLIP"):
            # Tokenize and encode text
            with torch.no_grad():
                tokens_t = clip.tokenize([t]).to(self.device)
                tokens_t_eng = clip.tokenize([t_eng]).to(self.device)
                embed_t = self.clip_model.encode_text(tokens_t).squeeze(0).to(torch.float32)  
                embed_t_eng = self.clip_model.encode_text(tokens_t_eng).squeeze(0).to(torch.float32)  

                embed_t = embed_t.unsqueeze(0).unsqueeze(0)  # Add batch and component dims: [1, 1, 512]
                embed_t_eng = embed_t_eng.unsqueeze(0).unsqueeze(0)  # [1, 1, 512]

                embed_t = self.cb.pre_encoder_bias(embed_t)  # Shape: [1, 1, 512]
                embed_t_eng = self.cb.pre_encoder_bias(embed_t_eng)  # Shape: [1, 1, 512]

                
                concept_embed_t = self.cb.encoder(embed_t).squeeze(0).squeeze(0) 
                concept_embed_t_eng = self.cb.encoder(embed_t_eng).squeeze(0).squeeze(0)

            self.embeddings_t.append(concept_embed_t.cpu())
            self.embeddings_t_eng.append(concept_embed_t_eng.cpu())

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):
        return self.embeddings_t[idx], self.embeddings_t_eng[idx]

# --- Define Adaptive Layer ---
class AdaptiveLayer(nn.Module):
    # in literature, also called Concept Intervention
    def __init__(self, input_dim=4096, output_dim=4096):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, output_dim),
            nn.LogSoftmax(dim=-1)  # For KL Divergence. We use this because we are mapping concepts
        )

    def forward(self, x):
        return self.fc(x)

# --- Training Loop ---
def train_mapping():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CLIP model
    clip_model, _ = clip.load("ViT-B/16", device=device)
    clip_model.eval()

    # Load dataset (current structure: [(t_elvish, t_english), ...])
    text_pairs = load_elvish_dataset()
    dataset = TextEmbeddingDataset(text_pairs, clip_model, device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize adaptive layer mapping concepts of elvish text to concepts of equivalent english text.
    mapping_net = AdaptiveLayer().to(device)
    optimizer = optim.Adam(mapping_net.parameters(), lr=1e-4)
    criterion = nn.KLDivLoss(reduction="batchmean")  # KL-Divergence

    # Training
    num_epochs = 20
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0.0
        mapping_net.train()
        for batch_t, batch_t_eng in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_t, batch_t_eng = batch_t.to(device), batch_t_eng.to(device)
            
            optimizer.zero_grad()
            pred_t_eng = mapping_net(batch_t)
            
            # Convert target embeddings to probabilities (softmax)
            target_probs = F.softmax(batch_t_eng, dim=-1)
            loss = criterion(pred_t_eng, target_probs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"mapping_net_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': mapping_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "clip_mapping_net_final_ViT-B-16.pth")
    torch.save(mapping_net.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

if __name__ == "__main__":
    train_mapping()