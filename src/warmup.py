# In this file, we define load_model
# It runs once at server startup to load the model to a GPU

## In this example: A Huggingface BERT model

# from transformers import T5ForConditionalGeneration
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from huggingface_hub import hf_hub_download
import clip


def load_model():

    # load the model from cache or local file to the CPU
    # model = T5ForConditionalGeneration.from_pretrained('t5-small').cuda()
    
    model1 = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer1 = GPT2Tokenizer.from_pretrained("gpt2")
    conceptual_weight = hf_hub_download(repo_id="akhaliq/CLIP-prefix-captioning-conceptual-weights", filename="conceptual_weights.pt")
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # transfer the model to the GPU
    # N/A for this example, it's already on the GPU

    # return the callable model
    return model1,tokenizer1,conceptual_weight,clip_model,preprocess
