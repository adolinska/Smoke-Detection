import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import gradio as gr
from huggingface_hub import hf_hub_download
import torch.nn as nn
import timm 

REPO_ID = "Raaniel/model-smoke"
MODEL_FILE_NAME = "best_model_epoch_32.pth"
USE_CUDA = torch.cuda.is_available()
num_classes = 3

# Download the model
checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE_NAME)

# Load the checkpoint
state = torch.load(checkpoint_path, map_location=torch.device('cuda' if USE_CUDA else 'cpu'))

# Create the model and modify it
model = timm.create_model('mobilenetv3_small_050', pretrained=True)
num_features = model.classifier.in_features

# Additional linear and dropout layers
model.classifier = nn.Sequential(
    nn.Linear(num_features, 256),  # Additional linear layer
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)  # Final classification layer
)

# Load the model weights
model.load_state_dict(state)

# Move model to the appropriate device
device = torch.device('cuda' if USE_CUDA else 'cpu')
model = model.to(device)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])
])

classes = ["clouds", 'other', "smoke"]

def predict(image, model=model, classes=classes, device=device, transform=transform):
    model.eval()

    print(type(image))
    # Check if the image is a PyTorch Tensor, if so, use it directly
    if isinstance(image, torch.Tensor):
        img_batch = image.unsqueeze(0).to(device)
    elif isinstance(image, np.ndarray):  # Check if the image is a numpy ndarray
        # Convert numpy ndarray to PIL Image
        img = Image.fromarray(image)
        # Transform the image
        img_transformed = transform(img)
        # Convert to a batch of 1 and send to device
        img_batch = img_transformed.unsqueeze(0).to(device)
    else:
        # Load the image and apply transformations
        img = Image.open(image)
        img_transformed = transform(img)
        img_batch = img_transformed.unsqueeze(0).to(device)

    # Make predictions
    with torch.no_grad():
        _, predicted_idx = model(img_batch).max(1)

    # Map the index to the class name
    predicted_class = classes[predicted_idx.item()]

    return predicted_class

examples = ["https://bi.im-g.pl/im/83/4d/1c/z29677955IEG,Chmury-pieknej-pogody-Cumulus-humilis-.jpg",
            "https://energyeducation.ca/wiki/images/5/51/Smoke_column_-_High_Park_Wildfire_%281%29.jpg",
            "https://thumb.bibliocad.com/images/content/00000000/9000/9813.jpg",
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRC7j2LoW8D13BOgbT_9J2SI_krX0sadT4oaSuyFjNb3jElJdU-J7DpPgCYvEfFzqoD6c0&usqp=CAU"]


css = """
h1 {
    text-align: center;
    display:block;
}
"""

with gr.Blocks(theme=gr.themes.Base(primary_hue="zinc",
                                    secondary_hue="neutral",
                                    neutral_hue="slate",
                                    font =  gr.themes.GoogleFont("Montserrat")),
               css = css,
               title="Smoke Detection") as demo:

    demo.load(None, None, js="""
  () => {
  const params = new URLSearchParams(window.location.search);
  if (!params.has('__theme')) {
    params.set('__theme', 'light');
    window.location.search = params.toString();
  }
  }""",
    )

    markdown_content = """
    <img src='file/dd_logo.png' width='200'>
    """
    gr.Markdown(markdown_content)
    gr.Markdown("# 🔥 Early Fire Detection 🔥")

    gr.Markdown(""" ## Spot Fire, Preserve Nature! Effortlessly tell apart smoke from clouds using our smart fire detection technology.
    Our system is enhanced by a comprehensive database of more than 14,000 images and sophisticated machine learning algorithms, 
    facilitating prompt identification of fire. Fast, intelligent, and vigilant – we safeguard our environment against the initial threat signs.
    
    
    The model was trained on the "smokedataset" by Jakub Szumny, from the Math and Computer Science Division at the University of Illinois at Urbana-Champaign. 
    This dataset is accessible at [Hugging Face](https://huggingface.co/datasets/sagecontinuum/smokedataset).""")

    with gr.Accordion("Details", open = False):
      gr.Markdown("""The rise in fire incidents, intensified by climate change, poses a significant challenge for quick detection and action.
    Conventional methods of fire detection, like manual observation and reporting, are often too slow, particularly in remote locations.
    Automated smoke detection systems provide a solution, leveraging deep learning for rapid and precise smoke detection in images.
    The skill to differentiate smoke from visually similar occurrences, such as clouds, is vital. This distinction leads to quicker identification of fire sources,
    allowing for faster response times and possibly preserving large tracts of natural and inhabited areas from devastation.
    Enhancing the speed and precision of fire detection can greatly reduce their effects on communities, economies, and ecosystems.""")

    with gr.Column():
      image = gr.Image(label = "Picture")

      gallery = gr.Gallery(value = examples, label="Example photos",columns=[4], rows=[1], height=200, object_fit = "scale-down")

      def get_select_index(evt: gr.SelectData):
        return examples[evt.index]

      gallery.select(get_select_index, None, image)
      action = gr.Button("Detect")
      prediction = gr.Textbox(label = "Prediction")

      action.click(fn=predict, inputs=image, outputs=prediction)

demo.launch(width = "75%", debug = True, allowed_paths=["/"], share = True) # add ?__theme=light to url to force lightmode