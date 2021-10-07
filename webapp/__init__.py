from flask.app import Flask
from PIL import Image
import torch
from torch import nn
from torchvision import models

from torchvision import transforms


def do_something():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to("cuda")

    model_ft.load_state_dict(torch.load("../bee-model.pth"))

    model_ft.eval()
    return model_ft


class MLFlask(Flask):
    model = do_something()

    def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):
        if not self.debug:
            with self.app_context():
                self.model = do_something()
        super(MLFlask, self).run(host=host, port=port,
                                 debug=debug, load_dotenv=load_dotenv, **options)


app = MLFlask(__name__)
app.run()
