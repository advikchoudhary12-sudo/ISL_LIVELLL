import torch

from isl_recognition.labels import STATIC_LABELS
from isl_recognition.models.static_cnn import StaticSignCNN


def test_static_model_input_output_shapes():
    model = StaticSignCNN(num_classes=len(STATIC_LABELS))
    output = model(torch.zeros(2, 3, 96, 96))
    assert output.shape == (2, len(STATIC_LABELS))
