import typer
from taxmusr.generate import generate_examples
from taxmusr.evaluate import run_evaluation


app = typer.Typer()

@app.command()
def generate(
        domain: str = typer.Option("joint_assessment", help="The domain to generate the dataset for."),
        num_samples: int = typer.Option(10, help="The number of samples to generate."),
        output_dir: str = typer.Option(..., help="The directory to output the generated dataset."),
        max_depth: int = typer.Option(2, help="The maximum depth for reasoning tree expansion."),
        model: str = typer.Option("openai:gpt-4o", help="The {model_provider}:{model} to use for generation."),
        temperature: float = typer.Option(1.0, help="The temperature to use for generation."),
        top_p: float = typer.Option(1.0, help="The top_p to use for generation."),
        max_tokens: int = typer.Option(2048, help="The maximum number of tokens to generate.")
):
    """
    Generate a dataset for the specified tax scenario.
    """
    print(f"Generating {num_samples} samples for scenario '{domain}' with max depth {max_depth} using model '{model}'")
    print(f"LLM parameters: temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}")
    _ = generate_examples(
        domain=domain,
        num_samples=num_samples,
        output_dir=output_dir,
        model=model,
        max_depth=max_depth,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )


@app.command()
def evaluate(
        dataset: str = typer.Option(..., help="The path to the dataset to evaluate."),
        output_path: str = typer.Option(..., help="The path to output the evaluation results."),
        model: str = typer.Option("openai:gpt-4o", help="The model to use for evaluation."),
        workflow: str = typer.Option("cot", help="The evaluation workflow to use."),
        temperature: float = typer.Option(1.0, help="The temperature to use for generation."),
        top_p: float = typer.Option(1.0, help="The top_p to use for generation."),
        max_tokens: int = typer.Option(2048, help="The maximum number of tokens to generate.")
):
    """
    Evaluate the specified dataset using the given GenAI workflow.
    """
    print(f"Evaluating dataset at '{dataset}' using model '{model}' using workflow '{workflow}'")
    run_evaluation(
        dataset=dataset,
        output_path=output_path,
        workflow=workflow,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )


if __name__ == "__main__":
    app()
