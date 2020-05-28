import torch

from src.models.onnx import SpatialVaeEncoder, SpatialVaeDecoder

def main():
  model = SpatialVaeEncoder()
  # pytorch_model.load_state_dict(torch.load('pytorch_model.pt'))
  model.eval()
  dummy_input = torch.zeros(28, 28)
  torch.onnx.export(model, dummy_input, 'onnx/svae_encoder.onnx', verbose=True)


  model = SpatialVaeDecoder()
  # pytorch_model.load_state_dict(torch.load('pytorch_model.pt'))
  model.eval()
  dummy_input = torch.zeros(784, 4)
  torch.onnx.export(model, dummy_input, 'onnx/svae_decoder.onnx', verbose=True)


if __name__ == '__main__':
  main()
