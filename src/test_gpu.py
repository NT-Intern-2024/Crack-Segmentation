import tensorflow as tf
import torch


def check_tensorflow_gpu():
    print("TensorFlow version:", tf.__version__)
    print("CUDA support built-in TensorFlow:", tf.test.is_built_with_cuda())

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("TensorFlow has found a GPU.")
        for gpu in gpus:
            print(f" - {gpu}")
            print("GPU details:", tf.config.experimental.get_device_details(gpu))
    else:
        print("TensorFlow did not find a GPU.")


def check_pytorch_gpu():
    print("PyTorch version:", torch.__version__)
    print("CUDA available in PyTorch:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("PyTorch has found a GPU.")
        print(f" - GPU: {torch.cuda.get_device_name(0)}")
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
    else:
        print("PyTorch did not find a GPU.")


def main():
    print("Checking TensorFlow...")
    check_tensorflow_gpu()
    print("\nChecking PyTorch...")
    check_pytorch_gpu()


if __name__ == "__main__":
    main()
