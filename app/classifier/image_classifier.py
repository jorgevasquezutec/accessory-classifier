import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torchvision




def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class ImageClassifier:
    def __init__(self, model_path,class_map_path,device='cpu'):
        self.device = device
        # Revisar si CUDA está disponible
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        
        # Cargar el modelo
        # self.model = torch.load(model_path, map_location=self.device)
        self.model = models.shufflenet_v2_x1_0(weights='ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1')
        for param in self.model.parameters():
            param.requires_grad = False

        with open(class_map_path) as f:
            self.classes = [line.strip() for line in f.readlines()]

      

        self.model.fc = nn.Linear(1024, len(self.classes))
        if torch.cuda.is_available():
            self.model = to_device(self.model, self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Configurar la transformación de las imágenes
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

       
            # print(self.classes)
    
    def predict(self, image_path):
        # Cargar la imagen
        image = Image.open(image_path)
        
        # Aplicar la transformación
        image_tensor = self.transform(image).unsqueeze(0)
        
        #Pasar la imagen por el modelo
        with torch.no_grad():
            if torch.cuda.is_available():
                output = self.model(image_tensor.to(self.device))
            else:
                output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
        
        # Obtener los índices de las tres clases con mayor probabilidad
        top3_prob, top3_idx = torch.topk(probabilities, k=3)

        # Obtener las clases correspondientes a los índices
        predicted_classes = [self.classes[idx] for idx in top3_idx.tolist()]

        # Obtener las probabilidades correspondientes a los índices
        predicted_probabilities = top3_prob.tolist()

        # Crear un diccionario con las clases y probabilidades
        result_dict = {}
        for i in range(len(predicted_classes)):
            result_dict[predicted_classes[i]] = predicted_probabilities[i]

        return result_dict
