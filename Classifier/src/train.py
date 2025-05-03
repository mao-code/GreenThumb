import argparse
from data_loader import create_generators
from model import build_model, build_transfer_model
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_model', default='cnn_model.h5')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--transfer', action='store_true', help='Use transfer learning')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size for VGG16 model')
    parser.add_argument('--trainable_blocks', nargs='*', default=None, help='Trainable blocks for transfer learning, like "block5_conv"')
    args = parser.parse_args()

    train_gen, val_gen, _ = create_generators(
        base_dir=args.data_dir,
        batch_size=args.batch_size
    )

    # Save class indices to a JSON file
    class_indices = train_gen.class_indices
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f)

    num_classes = train_gen.num_classes

    if args.transfer:
        model = build_transfer_model(
            image_size=args.image_size,
            num_classes=num_classes,
            trainable_blocks=args.trainable_blocks
        )
    else:
        model = build_model(num_classes=num_classes)

    model = build_model(num_classes=num_classes)

    # Model training
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs
    )

    model.save(args.output_model)
    print(f'Model has been saved to {args.output_model}')
    print('Class indices saved to class_indices.json')

if __name__ == '__main__':
    main()