import json
from pathlib import Path
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from taxmusr.workflows.base import BaselineWorkflow


def run_evaluation(
        dataset: str,
        output_path: str,
        workflow: str,
        model: str = "openai:gpt-4o",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
):
    """Evaluate the specified dataset using the given GenAI workflow.
    :param dataset: The path to the dataset to evaluate. Should be a .json or .jsonl file.
    :param output_path: The path to output the evaluation results. Will be a .jsonl file.
    :param workflow: The evaluation workflow to use. Currently only "cot" is supported.
    :param model: The model to use for evaluation.
    :param temperature: The temperature to use for generation.
    :param top_p: The top_p to use for generation.
    :param max_tokens: The maximum number of tokens to generate.
    """
    # Read dataset
    examples = []
    with open(dataset, "r", encoding="utf8") as f:
        if dataset.endswith(".json"):
            examples = json.load(f)
        elif dataset.endswith(".jsonl"):
            for line in f:
                examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples from {dataset}")

    # Set up GenAI workflow
    baseline = BaselineWorkflow(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        cot=True if workflow == "cot" else False,
    )

    # Evaluate each example
    y_true = []
    y_pred = []
    examples_with_predictions = []
    for example in tqdm(examples, desc="Evaluating examples"):
        y_true += [example["answer"]]
        output = baseline.run(example)
        predicted_answer = output.predicted_answer
        reasoning = output.reasoning
        token_usage = output.token_usage
        y_pred.append(predicted_answer)
        example["prediction"] = {
            "predicted_answer": predicted_answer,
            "reasoning": reasoning,
            "token_usage": token_usage
        }
        examples_with_predictions.append(example)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf8") as f:
        for example in examples_with_predictions:
            f.write(json.dumps(example) + "\n")
    print(f"Wrote {len(examples_with_predictions)} evaluated examples to {str(output_path)}")
