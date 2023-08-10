import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # Define a CNN architecture.
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1), # 224x224x3 -> 224x224x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),    
            
            nn.Conv2d(16, 32, 3, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56x56x32
            
            nn.Conv2d(32, 64, 3, padding=1),  
            nn.BatchNorm2d(64),
            #nn.Dropout2d(p=dropout),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # outputsize = 28x28x64
            
            nn.Conv2d(64, 128, 3, padding=1),  
            nn.BatchNorm2d(128),
            #nn.Dropout2d(p=dropout),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # outputsize = 14x14x128
            
            nn.Conv2d(128, 256, 3, padding=1),  
            nn.BatchNorm2d(256),
            #nn.Dropout2d(p=dropout),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7x7x256
            
            nn.Flatten(),  # -> 1x256X7X7
            
            nn.Linear(256 * 7 * 7 , 1024),  # -> 512
            nn.Dropout(p = dropout),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024 , 512),  # -> 512
            nn.Dropout(p = dropout),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
