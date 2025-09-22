# TaxMuSR: Multistep Soft Reasoning for Tax Law Cases

A Python implementation inspired by the MuSR (Multistep Soft Reasoning) paper, specifically designed for generating and evaluating synthetic tax law cases that require complex multi-step reasoning.

## Overview

This project was developed as part of a technical challenge to apply the principles from the ["MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning"](https://arxiv.org/abs/2310.16049) paper to the domain of tax law. This is a fresh, independent implementation that focuses specifically on German tax scenarios.

The original repository for the MuSR paper: https://github.com/Zayne-Sprague/MuSR

Here is the citation for the original MuSR paper:
```tex
@article{sprague2023musr,
  title={Musr: Testing the limits of chain-of-thought with multistep soft reasoning},
  author={Sprague, Zayne and Ye, Xi and Bostrom, Kaj and Chaudhuri, Swarat and Durrett, Greg},
  journal={arXiv preprint arXiv:2310.16049},
  year={2023}
}
```

## Features

- **Synthetic Tax Case Generation**: Creates tax scenarios with embedded reasoning chains
- **Multi-step Reasoning Trees**: Generates complex reasoning structures that require multiple logical steps
- **Domain-Specific Implementation**: Currently supports two tax domains:
  - Joint Assessment: Is joint assessment or individual assessment more beneficial? 
  - Home Office Deduction: Is a home office pro-rata deduction applicable or should the taxpayer claim the flatrate?
- **Evaluation Framework**: Built-in evaluation system for testing model performance on generated cases
- **LangChain Integration**: Utilizes LangChain for model interaction allowing for easy swapping of LLM providers
- **Extensible Architecture**: Modular design for adding new tax domains and reasoning workflows
- **CLI Interface**: Easy-to-use command-line interface for generation and evaluation

## Installation

### Prerequisites
- Python 3.11 or higher
- UV package manager (recommended)

### Setup

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd taxmusr
    ```
2. Set up venv and install dependencies using UV:
    ```bash
    uv venv --python=3.11
    uv sync
    ```
3. Set up environment variables (create a `.env` file based on `.env.example`):
    ```bash
    OPENAI_API_KEY=your_openai_api_key
    LANGFUSE_PUBLIC_KEY=your_langfuse_public_key  # Optional, for logging
    LANGFUSE_SECRET_KEY=your_langfuse_secret_key  # Optional, for logging
    ```
4. Install the `taxmusr` package in editable mode:
    ```bash
    uv pip install -e .
    ```
   
If you want to use other providers (e.g., Huggingface, Azure, Anthropic), make sure to set the corresponding 
environment variables and install the necessary LangChain integrations: https://python.langchain.com/docs/integrations/providers/


## Usage

The project provides a command-line interface (CLI) via `taxmusr/cli.py` for generating and evaluating tax cases.
The main commands are `generate` and `evaluate`.

### Generate Tax Cases

Generate synthetic tax cases for a specific domain:

```bash
uv run taxmusr generate \
    --domain joint_assessment \
    --num-samples 5 \
    --output-dir ./datasets \
    --max-depth 2 \
    --model "openai:gpt-4o" \
    --temperature 1.0
```

This writes the generated cases to `./datasets/joint_assessment_cases.jsonl`.

For details on generation arguments, you can run:
```bash
uv run taxmusr generate --help
```

You can find some data samples in the `datasets` folder.

### Evaluate Cases

Evaluate generated cases using different reasoning workflows:

```bash
uv run taxmusr evaluate \
    --dataset ./datasets/joint_assessment_cases.jsonl \
    --output-path ./results/joint_assessment_cases_results.jsonl \
    --model "openai:gpt-4o" \
    --workflow cot \
    --temperature 1.0
```

This writes the evaluation results to `./results/joint_assessment_cases_results.jsonl`.
The original examples are augmented with the predicted answers, reasoning and token usage.

For details on evaluation arguments, you can run:
```bash
uv run taxmusr evaluate --help
```

## Tax Domains

### Joint Assessment

Generates cases involving German couples deciding between joint and individual tax assessment. Cases consider factors such as:
- Income disparity between spouses
- Residence requirements
- Church tax implications for mixed-religion couples

### Home Office Deduction

Creates scenarios around home office deduction eligibility, considering:
- Workspace exclusivity requirements
- Business necessity criteria

### Adding your own domain

To add a new tax domain, create a new module in `taxmusr/domains` and implement the necessary components:
- `domain.py`: Define the domain-specific logic and rules
- `logic.py`: Optionally implement a rule engine for fact checking
- `rules.py`: Define a set of relevant rules/ heuristics that apply to the domain and can be used in reasoning
- `prompts.py`: Create prompts for case generation and reasoning

Update the imports in `taxmusr/generate.py` and domain mapping.

## Generated Case Format

Each generated case includes:
- **Narrative**: A first-person story describing the situation
- **Underlying Facts**: Key facts serving as the foundations for the narrative
- **Rule Signals**: Relevant tax/ commonsense rules that apply to the case
- **Reasoning Tree**: Multi-step logical reasoning structure
- **Question**: The tax decision to be made
- **Answer**: The correct answer
- **Reasoning Trace**: Step-by-step logical reasoning

## Approach
I started this project by forking the original repository and adapting it to the tax domain, but decided implementing it from scratch would be more efficient.
The basic approach still follows the principles from the MuSR paper, but with some simplifications and adaptations to fit the tax domain.
We start with a set of gold facts, expand them using a reasoning tree, and then generate a narrative that embeds the facts.
This kind of workflow can be seen in the abstract `TaxDomain` class in `taxmusr/domains/base.py`.

### Project Structure 

```
taxmusr/
├── taxmusr/                           # Main package
│   ├── cli.py                        # Command-line interface entry point
│   ├── generate.py                   # Generation framework
│   ├── evaluate.py                   # Evaluation framework and metrics
│   │
│   ├── core/                         # Core framework components
│   │   ├── schemas.py               # Pydantic data models and validation
│   │   ├── generator.py             # High-level case generation orchestration
│   │   ├── chat_model.py            # LLM interaction abstractions
│   │
│   ├── domains/                      # Tax domain implementations
│   │   ├── base.py                  # Abstract base class for tax domains
│   │   ├── formatter.py             # Common formatting utilities
│   │   │
│   │   ├── joint_assessment/        # Joint vs individual assessment domain
│   │   │   ├── domain.py           # Domain-specific case generation logic
│   │   │   ├── rules.py            # Tax rules and heuristics
│   │   │   ├── prompts.py          # LLM prompts for generation
│   │   │   ├── logic.py            # Tax logic and calculations
│   │   │
│   │   └── home_office_deduction/   # Home office deduction domain
│   │       ├── domain.py           # Domain-specific case generation logic
│   │       ├── rules.py            # Relevant tax rules
│   │       ├── prompts.py          # Generation prompts
│   │       ├── logic.py            # (Not implemented) Deduction calculations
│   │
│   └── workflows/                    # Evaluation workflows
│       ├── base.py                  # Base workflow implementation (CoT, Few-Shot)
│
├── datasets/                         # Generated datasets
├── results/                          # Evaluation results and analysis
├── resources/                        # Static resources (collection of rules)
├── tests/                           # Test suite
```

## TODOs
Given the time constraints, there are still many features and improvements to be made. Here are some of the planned TODOs:

- [ ] Validation and deduplication steps when building the reasoning trees (keyword checks, regex, fuzzy matching)
- [ ] Add more GenAI workflows (create training examples for few-shot, implement agentic workflows with tool use)
- [ ] Add more tax domains (continue working integrating more realistic rule engines `GroundedJointAssessmentDomain` is a start)
- [ ] Improve prompts and generation quality
- [ ] Test more models and providers (e.g., Azure, Anthropic)
- [ ] Add more complexity to the reasoning trees (e.g., more node types, deeper trees, pruning strategies)
- [ ] Implement caching for model calls to reduce costs and improve speed
- [ ] Add unit and integration tests
- [ ] Improve logging (using logging library/ rich library), print token usage and costs
- [ ] Switch to data format that is compatible with the original MuSR dataset for easier comparison/ interoperability