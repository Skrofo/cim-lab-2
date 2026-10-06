import torch
from types import SimpleNamespace
from cim_lab_2.models import build_model, configure_trainable_parameters, parameter_counts


def make_model():
    return build_model(SimpleNamespace(pretrained=False, num_classes=37))


def test_output_shape():
    model = make_model()
    output = model(torch.zeros(2, 3, 224, 224))
    assert output.shape == (2, 37)


def test_linear_probe_only_fc_trainable():
    model = make_model()
    configure_trainable_parameters(model, "linear_probe")
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert names and all(name.startswith("fc.") for name in names)


def test_partial_unfreezes_layer4_and_fc():
    model = make_model()
    configure_trainable_parameters(model, "partial")
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert any(name.startswith("layer4.") for name in names)
    assert any(name.startswith("fc.") for name in names)
    assert not any(name.startswith("layer3.") for name in names)


def test_full_unfreezes_everything():
    model = make_model()
    configure_trainable_parameters(model, "full")
    counts = parameter_counts(model)
    assert counts["parameters_total"] == counts["parameters_trainable"]
