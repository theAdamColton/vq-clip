import torch
from transformers import CLIPProcessor
from vq_clip import VQCLIPModel
from vq_clip.eval import zero_shot_eval


def evaluate(
        imagenet_path: str = "",
        pretrained_clip_url: str = "openai/clip-vit-large-patch14",
        vq_vision_model_url: str = "adams-story/vq-ViT-L-14-k64-d32",
        batch_size: int = 1024,
        ):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = VQCLIPModel.from_pretrained_clip(pretrained_clip_url, vision_vq_adapter_path=vq_vision_model_url)
    print("loaded vqclip")

    processor = CLIPProcessor.from_pretrained(pretrained_clip_url)

    model = model.to(device)

    with torch.no_grad():
        with torch.autocast(str(device)):
            res = zero_shot_eval(model, processor, imagenet_path, batch_size = batch_size)
            print(res)
    

if __name__ == "__main__":
    import jsonargparse
    jsonargparse.CLI(evaluate)
