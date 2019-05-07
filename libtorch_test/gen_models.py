import torch
import torchvision

# A sample model
model = torchvision.models.resnet18()
# An example input you will provide to your model. (must match exact dimension)
example = torch.rand(1, 3, 224, 224)
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
# output = traced_script_module(torch.ones(1, 3, 224, 224))
traced_script_module.save("model.pt")
