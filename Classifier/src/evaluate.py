import argparse
from tensorflow.keras.models import load_model
from data_loader import create_generators

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    _, _, test_gen = create_generators(
        base_dir=args.data_dir,
        batch_size=args.batch_size
    )

    # Load the model
    model = load_model(args.model_path)

    loss, acc = model.evaluate(test_gen)
    print(f'Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')

if __name__ == '__main__':
    main()