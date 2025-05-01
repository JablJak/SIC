import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math

# Załóżmy, że obrazy są w folderze 'path/to/your/dataset'
# Ważne: Użyj transformacji tylko do konwersji na tensor i skalowania do [0, 1]
# Nie używaj jeszcze normalizacji!
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Opcjonalnie, jeśli obrazy mają różne rozmiary
    transforms.ToTensor()  # Konwertuje obraz (PIL/numpy) [0, 255] HWC do tensora [0, 1] CHW
])

dataset = datasets.ImageFolder(root='/run/media/jakub/Dane/Studia/INZ/data/coco', transform=transform)
# Użyj DataLoader do efektywnego ładowania danych
# batch_size może być większy dla przyspieszenia, ale uważaj na pamięć RAM
loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=4)

mean = torch.zeros(3)
sum_sq = torch.zeros(3)
num_pixels = 0
print(len(loader))
for index, (images, _) in enumerate(loader):
    print(index)
    # images ma kształt [batch_size, channels, height, width]
    batch_size, channels, height, width = images.shape

    # Oblicz liczbę pikseli w tej paczce
    batch_pixels = batch_size * height * width
    num_pixels += batch_pixels

    # Oblicz sumę wartości pikseli dla każdego kanału w tej paczce
    # Sumuj po wymiarach: batch(0), height(2), width(3)
    mean += images.sum([0, 2, 3])

    # Oblicz sumę kwadratów wartości pikseli dla każdego kanału w tej paczce
    sum_sq += (images ** 2).sum([0, 2, 3])

# Oblicz końcową średnią
mean /= num_pixels

# Oblicz końcową średnią kwadratów
sum_sq /= num_pixels

# Oblicz wariancję: Var = E[X^2] - (E[X])^2
var = sum_sq - (mean ** 2)

# Oblicz odchylenie standardowe
std = torch.sqrt(var)

print(f"Całkowita liczba pikseli: {num_pixels}")
print(f"Obliczona średnia (mean): {mean}")
print(f"Obliczona wariancja (variance): {var}")
print(f"Obliczone odchylenie standardowe (std): {std}")

# Przykład użycia w transformacjach normalizujących:
# normalize_transform = transforms.Normalize(mean=mean.tolist(), std=std.tolist())