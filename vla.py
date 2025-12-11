# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import pillow_heif
import torch

pillow_heif.register_heif_opener()

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("mps")

path = "/Users/mathisscheffler/Desktop/pinocchio-minimal-main/attention.HEIC"
path = "/Users/mathisscheffler/Desktop/pinocchio-minimal-main/close gripper.HEIC"
path = "/Users/mathisscheffler/Desktop/pinocchio-minimal-main/image_task_grab.HEIC"
path = "/Users/mathisscheffler/Desktop/pinocchio-minimal-main/noobstacle.HEIC"
image: Image.Image = Image.open(path)
prompt = "In: What action should the robot take to grab the cube?\nOut:"

inputs = processor(prompt, image).to("mps", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(action)
