from tqdm.auto import tqdm
import polars as pl
from pathlib import Path
import json
import torch
import click
from src.classifiers.zeroshot import ZeroShotClassifierWithProbs, get_system_prompt


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FOLDER = BASE_DIR / "data"


@click.command()
@click.option("--model_name", default="Qwen/Qwen3-1.7B", help="Model name")
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default="cuda",
    help="Device to use for inference",
)
def main(model_name: str, device: str):
    df = pl.read_json(DATA_FOLDER / "generated_data.json")
    zero_shot_classifier = ZeroShotClassifierWithProbs(
        system_prompt=get_system_prompt(),
        model_name=model_name,
        device=device,
    )
    prompt_template = """
    Classify the command as malicious or not:
    command: {command}
    duration: {duration}
    exit_code: {exit_code}
    cwd: {cwd}
    """

    # вычисляем предсказания
    predictions = []
    for row in tqdm(df.iter_rows(named=True), total=len(df)):
        prompt = prompt_template.format(**row)
        pred_prob = zero_shot_classifier.classify(
            prompt=prompt,
            target_tokens={"pos": "Yes", "neg": "No"},
            use_chat_template=True,
            do_normalization=True,
            debug=False,
        )
        row["predicted_prob"] = pred_prob
        predictions.append(row)

    output_file = DATA_FOLDER / "predictions.json"
    if output_file.exists():
        with open(output_file, "r") as f:
            existing_predictions = json.load(f)
        existing_predictions.extend(predictions)
        predictions = existing_predictions
    with open(DATA_FOLDER / "predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
    return


if __name__ == "__main__":
    main()
