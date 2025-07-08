from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from diffusers import StableDiffusionPipeline

# Used for decoding the concept bottleneck to Text
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
sd_img_generator = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")




def get_prompt_text(concepts):
    decoder_input = "Concepts: " + ",".join(concepts)
    input_ids = tokenizer.encode(decoder_input, return_tensors = "pt")
    outputs = model.generate(input_ids, max_length=20, num_beams=4, early_stopping=True)
    prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prompt

def generate_image_via_SD(prompt):
    image = sd_img_generator(prompt).images[0]
    image.save("test.png")


# TODO Nima: Please train:
# 1. a concept bottleneck and 
# 2. The adaptive layer which maps elvish concepts to english concepts 

# elvish_input_text = "TODO: ADD STRINGS HERE"
# elvish_concept_vector = cbae(elvish_input_text)
# concept_vector = adaptive_layer(elvish_concept_vector)
# For proof of concept, consider the following concept vector: 
concept_vector = {"horse": 0.6, "garden": 1.8, "blue": -4, "grey": 1}
active_concepts = [c for c, v in concept_vector.items() if v > 0]
generate_image_via_SD(get_prompt_text(active_concepts))