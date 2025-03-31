import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Modell und Daten laden
model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Aktivierungen sammeln
activations = []

def get_activation(name):
        def hook(model, input, output):
            activations.append(output.detach())
            return hook

hook = model.layer4[1].register_forward_hook(get_activation('layer4'))

# Vorwärtsdurchlauf
for images, _ in dataloader:
    with torch.no_grad():
        model(images)

hook.remove()

# Aktivierungen verarbeiten
activations = torch.cat(activations).numpy().reshape(-1, 512) # Beispiel für ResNet18

# Dimensionalitätsreduktion
tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(activations)

# Visualisierung
plt.scatter(reduced[:, 0], reduced[:, 1])
plt.title('Activation Atlas')
plt.show()