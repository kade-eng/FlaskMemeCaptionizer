import warnings
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, url_for
import openai

#temporarily supress depricated warnings
warnings.filterwarnings("ignore")

#init Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    def generate_meme_caption(image_description):
        prompt = f"Generate a short funny 'meme' caption for a picture of '{image_description}'"
        
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50,  #length
            temperature=0.5  #creativity
        )
        
        caption = response.choices[0].text.strip()

        print("Generated Prompt:", prompt)
        print("Generated Caption: [" + caption + "]")
        return "\n" + caption
    
    def overlay_text_on_image(image, text):
        draw = ImageDraw.Draw(image)
        
        font_size = int(image.height * 0.055) 
        font = ImageFont.truetype("font/BodoniflfBold-MVZx.ttf", size=font_size)

        #draw and save image
        text_width, text_height = draw.textsize(text, font=font)
        position = ((image.width - text_width) // 2, image.height - text_height - 20) 
        draw.text(position, text, fill="white", font=font)
        
        image.save("static/output_image.jpg") 

    if 'image' not in request.files:
        return 'No image uploaded!', 400

    #load pre-trained resnet model
    resnet = models.resnet50(pretrained=True)
    resnet.eval()

    #define transformation (need to normalize img)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #pytorch needs to normalize with these values
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    #load + preprocess image
    image = request.files['image']
    image = Image.open(image)
    backup = image.copy()
    image = image.convert('RGB') #ensure input img had 3 channels before transformation
    image = transform(image)
    image = torch.unsqueeze(image, 0)  # Add an extra dimension for batch size

    #put image through resnet model
    with torch.no_grad():
        outputs = resnet(image)

    #get top 3 results
    result = "\n"
    name = ""
    _, top_predic = torch.topk(outputs, k=3)
    for i, idx in enumerate(top_predic.squeeze()):
        if i == 0:
            name = imagenet_classes[idx.item()]
        predicted_label = imagenet_classes[idx.item()]
        prob = torch.nn.functional.softmax(outputs, dim=1)[0][idx.item()].item()
        result = result + "Guess #" + str(i+1) + ": " + predicted_label + " (Confidence: " + str(round((prob*100),2)) + "%)\n"

    #meme the predictions
    openai.api_key = 'YOURAPIKEYHERE'

    # Example usage
    image_description = "a " + name
    cap = generate_meme_caption(image_description)
    overlay_text_on_image(backup, cap)

    return render_template('index.html', predicted_label=cap)

#main
if __name__ == '__main__':
    # Load ImageNet class labels
    with open('static/imagenet-classes.txt', 'r') as f:
        imagenet_classes = [line.strip() for line in f.readlines()]

    app.run(debug=True)