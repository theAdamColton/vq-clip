# VQ-CLIP

Finetune a CLIP model with a vector quantization bottleneck layer over the output embeddings. The quantization step is only applied to the final normalized CLIP embedding, and can be trained on a dataset of frozen CLIP embeddings.

# Pretrained VQ-CLIP models

### On top of openai/ViT-L-14

Both of these models were trained for roughly one epoch on datacomp medium, with a batch size of 16384. See `training_conf/VQ-ViT-L-14.yaml` for the training parameters that were used.

* [k=64 32 heads, multiheaded vq](https://huggingface.co/adams-story/vq-ViT-L-14-k64-d32-ema/tree/main): Gets 0.642 @1 on imagenet. Trained with EMA codebook rather than learnable. 

* [k=32 32 heads, residual quantization](https://huggingface.co/adams-story/vq-ViT-L-14-k32): Gets 0.51 @1 on imagenet validation. 

* [k=64 32 heads, vq quantization with affine parameters](https://huggingface.co/adams-story/vq-ViT-L-14-k64-d32): Gets 0.586 @1 on imagenet validation

# Set up env

```
$ conda create -n vq-clip
$ conda activate vq-clip
$ conda install pip -y
$ pip install -r requirements.txt
```

# Load a pretrained model

This will print a bunch of lines to the console complaining about missing `clip_model` weights in the state dict. Don't worry about it; the clip weights are loaded from `clip_path` argument.

```python
from PIL import Image
import requests
from vq_clip import VQCLIPModel
from transformers import CLIPProcessor

model = VQCLIPModel.from_pretrained_clip(clip_path="openai/clip-vit-large-patch14", vision_vq_adapter_path="adams-story/vq-ViT-L-14-k32", )

# make prediction
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
print(probs)
codes = outputs.image_codes # the vq codes
```


# Set up training data

You can train VQ-CLIP from a dataset of text-image CLIP embeddings. You can find these on [HuggingFace](https://huggingface.co/mlfoundations), I'd recommend using the image/text embeddings from  [datacomp 1B](mlfoundations/datacomp_1b) dataset. 

Only the .npx files are needed, these can be downloaded using the huggingface `snapshot_download` function.

This code downloads the dataset into the current directory:

```python
import sys
from huggingface_hub import snapshot_download
size = 'medium'
assert size in {"small", "medium", "large", "xlarge"}

snapshot_download(repo_id=f"mlfoundations/datacomp_{size}", repo_type="dataset", cache_dir="./hf-cache", local_dir=f"./{size}/metadata/", local_dir_use_symlinks=True, resume_download=True, allow_patterns="*.npz", max_workers=4)

print("\ndone.")
```

You can manually cut a single npx file from the downloaded data to be used as the validation set.

# Training

```
python train_rqclip_from_preembed.py fit -c conf/VQ-ViT-L-14.yaml --data.path_train /path/to/size/metadata/ --data.path_val /path/to/validation/metadata/ --model.vq_clip_config_path model_conf/vq-ViT-L-14-k1024/config.json
```

By default, training uses ~7GB VRAM, and saves a checkpoint and evaluates every 1000 steps

Training output is saved in the `out/` folder and can be viewed using tensorboard.

# ImageNet evaluation

* Download and extract imagenet 2012 val folder: https://academictorrents.com/details/207ebd69f80a3707f035cd91a114466a270e044d

* Change the folder structure into a format suitable for pytorch ImageFolder using [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh): 

* Run evaluation:

```python
from vq_clip import VQCLIPModel
from transformers import CLIPProcessor
from vq_clip.eval import zero_shot_eval

model = VQCLIPModel.from_pretrained_clip(clip_path="openai/clip-vit-large-patch14", vision_vq_adapter_path="adams-story/vq-ViT-L-14-k32", )
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

with torch.no_grad():
    with torch.autocast(device):
        top1, top5 = zero_shot_eval(vq_clip, processor, imagenet_path, validation_batch_size)
print(top1, top5)
```
