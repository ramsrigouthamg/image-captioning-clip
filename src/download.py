# In this file, we define download_model
# It runs during container build time to get model weights locally

# In this example: A Huggingface BERT model

# from transformers import T5ForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from huggingface_hub import hf_hub_download
import clip

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    # model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model1 = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer1 = GPT2Tokenizer.from_pretrained("gpt2")
    conceptual_weight = hf_hub_download(repo_id="akhaliq/CLIP-prefix-captioning-conceptual-weights", filename="conceptual_weights.pt")
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

if __name__ == "__main__":
    download_model()
