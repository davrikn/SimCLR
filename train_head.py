from classification_head import ClassificationHead
from load_predicted import load

x, y = load()

net = ClassificationHead()
net.to('cuda')

net.fit(x, y)

net.save()