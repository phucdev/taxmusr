from typing import List
from pathlib import Path

from taxmusr.core.generator import CaseGenerator
from taxmusr.core.schemas import GeneratedCase
from taxmusr.domains.joint_assessment.domain import JointAssessmentDomain
from taxmusr.domains.home_office_deduction.domain import HomeOfficeDeductionDomain


def generate_examples(
        domain: str,
        num_samples: int,
        output_dir: str = None,
        max_depth: int = 2,
        model: str = "openai:gpt-4o",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
) -> List[GeneratedCase]:
    """Generates a list of tax cases for the specified domain.
    :param domain: The tax domain to generate cases for. Currently only "joint_assessment" is supported.
    :param num_samples: The number of cases to generate.
    :param output_dir: The directory to output the generated cases to. If None, cases are not saved to disk.
    :param max_depth: The maximum depth for reasoning tree expansion.
    :param model: The {model_provider}:{model} to use for generation.
    :param temperature: The temperature to use for generation.
    :param top_p: The top_p to use for generation.
    :param max_tokens: The maximum number of tokens to generate.
    :return: A list of GeneratedCase objects.
    """
    if domain == "joint_assessment":
        domain_obj = JointAssessmentDomain(max_depth=max_depth)
    elif domain == "home_office_deduction":
        domain_obj = HomeOfficeDeductionDomain(max_depth=max_depth)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    generator = CaseGenerator(
        domain=domain_obj,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    generated_cases: List[GeneratedCase] = generator.generate(num_samples)
    print(f"Generated {len(generated_cases)} tax cases")
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir.joinpath(f"{domain_obj.name}_cases.jsonl")
        existing_cases = []
        if output_path.exists():
            with open(output_path, "r", encoding="utf8") as f:
                for line in f:
                    existing_cases.append(GeneratedCase.model_validate_json(line))
            print(f"Found {len(existing_cases)} existing cases from {str(output_path)}. Appending new cases.")
        with open(output_path, "w", encoding="utf8") as f:
            for case in existing_cases:
                f.write(case.model_dump_json() + "\n")
            for case in generated_cases:
                f.write(case.model_dump_json() + "\n")
        print(f"Wrote {len(generated_cases)} tax cases to {str(output_path)}")
    return generated_cases
