from tqdm.auto import tqdm
import polars as pl
from pathlib import Path
import json
import click
from src.classifiers.zeroshot import ZeroShotClassifierWithProbs, get_system_prompt


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FOLDER = BASE_DIR / "data"


@click.command()
@click.option(
    "--model-name", default="Qwen/Qwen3-1.7B", help="Model name, e.g., Qwen/Qwen3-1.7B"
)
@click.option("--input-file", help="Path to the input JSON file")
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default="cuda",
    help="Device to use for inference",
)
def main(model_name: str, device: str, input_file: str):
    input_file = Path(input_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    if input_file.suffix != ".json":
        raise ValueError(f"Input file {input_file} must be a JSON file.")

    df = pl.read_json(input_file)
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

    output_file = input_file.parent / "predictions.json"
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)
    return


if __name__ == "__main__":
    main()
