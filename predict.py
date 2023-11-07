import numpy as np
import torch

from load_predicted import load
from classification_head import ClassificationHead
from tqdm import tqdm


model = ClassificationHead()
model.load_state_dict(torch.load("classification_head_CIFAR.pth"))
model.to('cuda')

x, y = load()
predictions = list()
with torch.no_grad():
    for i in tqdm(range(len(x))):
        _x = torch.tensor(x[i]).to("cuda")
        _x = model(_x)
        predictions.append(np.argmax(_x.cpu().detach().numpy()))

correct = 0
for i in range(len(predictions)):
    if predictions[i] == y[i]:
        correct += 1

print(f"Accuracy: {correct/len(predictions)}")

