import argparse
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="tianchi-nlp-news-classification entrypoint")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("train", help="Train model using train.py")
    sub.add_parser("infer", help="Run inference using infer.py")

    args, extra = parser.parse_known_args()
    if args.cmd == "train":
        sys.argv = [sys.argv[0]] + extra
        import train as _train
        _train.main()
    elif args.cmd == "infer":
        sys.argv = [sys.argv[0]] + extra
        import infer as _infer
        _infer.main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
