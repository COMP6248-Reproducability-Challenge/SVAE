import torch

from src.models.onnx import SpatialVaeEncoder, SpatialVaeDecoder

def main():
  trained = torch.load('model_logs/mnist_rotated_translated_svae11_2.pt', map_location=torch.device('cpu'))
  # trained = torch.load('model_logs/mnist_svae11_2.pt', map_location=torch.device('cpu'))

  encoder = SpatialVaeEncoder()
  encoder_trained = {k: v for k, v in trained.items() if k in encoder.state_dict()}
  encoder.load_state_dict(encoder_trained)
  encoder.eval()
  dummy_input = torch.zeros(280 * 280 * 4)
  torch.onnx.export(encoder, dummy_input, 'web-res/svae_encoder.onnx', verbose=True)


  decoder = SpatialVaeDecoder()
  decoder_trained = {k: v for k, v in trained.items() if k in decoder.state_dict()}
  decoder.load_state_dict(decoder_trained)
  decoder.eval()
  dummy_input = torch.zeros(784, 4)
  torch.onnx.export(decoder, dummy_input, 'web-res/svae_decoder.onnx', verbose=True)


if __name__ == '__main__':
  main()
