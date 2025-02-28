import numpy as np
import torch

from load_embeddings import load
from classification_head import ClassificationHead
from tqdm import tqdm

from utils.arguments import get_config, get_arguments
from utils.utils import path_from_config

def predict(model, x, y):
    model.to('cuda')

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

    print(f"Accuracy: {correct / len(predictions)}")


if __name__ == "__main__":
    config = get_config(get_arguments())
    save_path = path_from_config(config)

    x, y = load(save_path)
    model = ClassificationHead(config)

    model.load_state_dict(torch.load(f"./models/classification_head{save_path}.pth"))

    predict(model, x, y)