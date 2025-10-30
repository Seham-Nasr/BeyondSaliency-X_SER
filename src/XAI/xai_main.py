from XAI.xAIutils import *
from XAI.exp_analysis import *
import itertools
import argparse


'''
Example usage:
python3 xai_main.py \
    --data_setname Crema-D \
    --XAI_method GradCAM \
    --Emotion angry

python -m XAI.xai_main --data_setnames Crema-D --XAI_methods GradCAM --emotions angry
python -m XAI.xai_main --data_setnames TESS --XAI_methods GradCAM --emotions angry

'''

def main():
    parser = argparse.ArgumentParser(description="Run XAI explanation analysis for SER datasets.")
    parser.add_argument(
        "--data_setnames",
        nargs="+",
        type=str,
        required=True,
        help="List of dataset names (e.g., Crema-D RAVDESS TESS)."
    )
    parser.add_argument(
        "--XAI_methods",
        nargs="+",
        type=str,
        required=True,
        help="List of XAI methods (e.g., GradCAM Occlusion CRP)."
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        type=str,
        required=True,
        help="List of emotions to analyze (e.g., happy sad neutral)."
    )

    args = parser.parse_args()

    # Iterate through all combinations of dataset, XAI method, and emotion
    for data_setname, XAI_method, Emotion in itertools.product(
        args.data_setnames, args.XAI_methods, args.emotions
    ):
        print("=========================================")
        print(f"Dataset: {data_setname}")
        print(f"XAI Method: {XAI_method}")
        print(f"Emotion: {Emotion}")
        print("=========================================")

        explanation_analysis(data_setname, XAI_method, Emotion)


if __name__ == "__main__":
    main()