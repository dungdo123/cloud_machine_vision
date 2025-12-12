# Model Directory

Place your trained model file here:

- **ONNX format**: `model.onnx`
- **TorchScript format**: `model.pt`

## Converting Models

### From PyTorch to ONNX

```python
import torch
import torch.onnx

# Load your model
model = YourModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["anomaly_score", "anomaly_map"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=14,
)
```

### From PyTorch to TorchScript

```python
import torch

# Load your model
model = YourModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Convert to TorchScript
scripted_model = torch.jit.script(model)
# or
scripted_model = torch.jit.trace(model, torch.randn(1, 3, 256, 256))

scripted_model.save("model.pt")
```

## Model Requirements

Your model should:
1. Accept input shape: `[batch, channels, height, width]` (NCHW format)
2. Output anomaly score (single value or max of anomaly map)
3. Optionally output anomaly map for visualization

## Expected Outputs

For anomaly detection models:
- `anomaly_score`: Single float value (0.0 - 1.0)
- `anomaly_map`: Optional heatmap for visualization

