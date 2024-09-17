from PIL import Image
from torchvision import transforms
import torch
import wget
import os


def image_object_identification(directory):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
    model.eval()
    for file in os.listdir(directory):
        if file.endswith('.png'):
            print(file)
            filename = os.path.join(directory, file)
            input_image = Image.open(filename)
            input_image = input_image.convert('RGB')
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')

            with torch.no_grad():
                output = model(input_batch)
            # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
            print(output[0])
            # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            print(probabilities)
            # Download ImageNet labels
            wget.download('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt')
            # Read the categories
            with open("imagenet_classes.txt", "r") as f:
                categories = [s.strip() for s in f.readlines()]
            # Show top categories per image
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            for i in range(top5_prob.size(0)):
                print(categories[top5_catid[i]], top5_prob[i].item())


directory = 'directory'
image_object_identification(directory)
