from utils.argutils import print_args
from encoder.evaluate import evaluate
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates the speaker encoder. You must have run encoder_preprocess.py first.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("run_id", type=str, help= \
        "Name for this model instance.")
    parser.add_argument("--val_root", type=Path, help= \
        "Path to the validation directory of encoder_preprocess.py.")
    parser.add_argument("-m", "--models_dir", type=Path, default="encoder/saved_models/", help= \
        "Path to the output directory that will contain the saved model weights, as well as "
        "backups of those weights and plots generated during training.")
    parser.add_argument("-n", "--num_iters", type=int, default=6, help= \
        "Number of validation steps.")
    args = parser.parse_args()

    # Process the arguments
    args.models_dir.mkdir(exist_ok=True)

    # Run the training
    print_args(args, parser)
    evaluate(**vars(args))
