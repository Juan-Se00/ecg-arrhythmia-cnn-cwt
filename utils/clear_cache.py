"""Free GPU memory immediately. Useful between hyperparameter trials."""
import torch


def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU cache cleared.")
    else:
        print("CUDA not available; nothing to clear.")


if __name__ == '__main__':
    main()
