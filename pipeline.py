from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import torch.nn as nn
import clip
from diffusers import StableDiffusionPipeline

from sparse_autoencoder import SparseAutoencoder

from train_adaptive_layer import AdaptiveLayer

# Used for decoding the concept bottleneck to Text
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
sd_img_generator = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None).to("cuda")



def get_prompt_text(concepts):
    decoder_input = "Concepts: " + ",".join(concepts)
    input_ids = tokenizer.encode(decoder_input, return_tensors = "pt")
    outputs = model.generate(input_ids, max_length=20, num_beams=4, early_stopping=True)
    prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prompt

def generate_image_via_SD(prompt):
    image = sd_img_generator(prompt).images[0]
    image.save("test.png")

class ConceptToTextEmbedding(nn.Module):
    def __init__(self, input_dim=4096, output_seq_len=77, embed_dim=768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_seq_len * embed_dim),
        )

    def forward(self, x):
        # x: [1, 4096] → [1, 77, 768]
        out = self.fc(x)
        return out.view(x.size(0), 77, 768)

 
# Loading Concept BottleNack
device = "cuda" if torch.cuda.is_available() else "cpu"
cb = SparseAutoencoder(
                n_input_features=512,
                n_learned_features=4096,
                n_components=1)

cb.load_state_dict(torch.load("./DNCBM/Checkpoints/clip_ViT-B_16_sparse_autoencoder_final.pt"))
cb.to(device)
cb.eval()


adaptive_layer = AdaptiveLayer(input_dim=4096, output_dim=4096).to(device)
adaptive_layer.load_state_dict(torch.load("./checkpoints/clip_mapping_net_final_ViT-B-16.pth"))
adaptive_layer.to(device)
adaptive_layer.eval()

clip_model, _ = clip.load("ViT-B/16", device=device)
clip_model.eval()



elvish_input_text = "Tára Aran Morion, colla mi telumë mornië calca, coronya angaina, henion carnë, ráma raumo mi má, turëa Grond. Nuru nírë sanda talan, nárë ar lómë órenyallo." # A towering Lord of Darkness, clad in glittering black armor, with an iron crown, red eyes, a storm-wing in hand, wielding Grond. Death’s chill on the broken plain, fire and night from his heart.
tokens_t = clip.tokenize([elvish_input_text]).to(device)
embed_t = clip_model.encode_text(tokens_t).squeeze(0).to(torch.float32)

learned_activations, decoded_activations = cb(embed_t)

concept_vector = adaptive_layer(learned_activations)

embedding_mapper = ConceptToTextEmbedding().to("cuda")

with torch.no_grad():
    custom_embedding = embedding_mapper(concept_vector)

image = sd_img_generator(
    prompt = None,
    prompt_embeds=custom_embedding,       # [1, 77, 768]
    num_inference_steps=100,
    guidance_scale=7.5,
    generator=torch.manual_seed(42)  
).images[0]    

image.save("morgoth1.png")

image2 = sd_img_generator(
    prompt = "A towering Lord of Darkness, clad in glittering black armor, with an iron crown, red eyes, a storm-wing in hand, wielding Grond. Death’s chill on the broken plain, fire and night from his heart.",
    num_inference_steps=100,
    guidance_scale=7.5,
    generator=torch.manual_seed(42)  
).images[0]
image2.save("morgoth2.png")