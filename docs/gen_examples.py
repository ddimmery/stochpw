"""Generate markdown documentation from examples with plots."""

import os
import subprocess
import sys
from pathlib import Path


def extract_key_code_from_file(content: str) -> str:
    """Extract the key_code string variable from an example file."""
    # Look for key_code = """...""" pattern
    import re

    # Match key_code = """...""" or key_code = '''...'''
    pattern = r'key_code\s*=\s*"""(.*?)"""'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        return match.group(1).strip()

    # Try single quotes
    pattern = r"key_code\s*=\s*'''(.*?)'''"
    match = re.search(pattern, content, re.DOTALL)

    if match:
        return match.group(1).strip()

    return "# Key code not found in example file"


def run_example_and_capture(example_path: Path, output_dir: Path, figures_dir: Path) -> dict:
    """
    Run an example script and capture its output.

    Returns a dict with 'stdout' and any generated plots.
    """
    # Change to output directory so plots are saved there
    original_dir = os.getcwd()
    os.chdir(output_dir)

    try:
        # Run the example
        result = subprocess.run(
            [sys.executable, str(example_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            print(f"Error running {example_path.name}:")
            print(result.stderr)
            return None

        output = {"stdout": result.stdout, "plots": []}

        # Find any generated PNG files
        png_files = list(Path(".").glob("*.png"))

        # Move plots to figures directory
        for png_file in png_files:
            dest = figures_dir / png_file.name
            png_file.rename(dest)
            output["plots"].append(dest.name)

        return output

    finally:
        os.chdir(original_dir)


def generate_markdown(example_name: str, output: dict, example_path: Path) -> str:
    """Generate markdown content from example output."""

    # Read the example file
    with open(example_path) as f:
        content = f.read()

    # Extract docstring if it exists
    description = ""
    if content.startswith('"""') or content.startswith("'''"):
        quote = '"""' if content.startswith('"""') else "'''"
        end_idx = content.find(quote, 3)
        if end_idx > 0:
            description = content[3:end_idx].strip()

    # Extract key_code from the example file
    key_code = extract_key_code_from_file(content)

    # Start markdown
    title = example_name.replace("_", " ").title()
    md = f"# {title}\n\n"

    if description:
        md += f"{description}\n\n"

    # Add key code section
    md += "## Code\n\n"
    md += "```python\n"
    md += key_code
    md += "\n```\n\n"

    # Add output in code block
    md += "## Output\n\n"
    md += "```\n"
    md += output["stdout"]
    md += "```\n\n"

    # Add plots if any
    if output["plots"]:
        md += "## Visualizations\n\n"
        for plot_name in output["plots"]:
            # Create a nicer caption from filename
            caption = plot_name.replace("_", " ").replace(".png", "").title()
            md += f"### {caption}\n\n"
            md += f"![{caption}](figures/{plot_name})\n\n"

    # Add expandable full source code with GitHub link
    github_url = f"https://github.com/ddimmery/stochpw/blob/main/examples/{example_path.name}"
    md += '??? example "Full source code"\n\n'
    md += "    ```python\n"
    # Indent each line of the full source
    for line in content.split("\n"):
        md += f"    {line}\n"
    md += "    ```\n\n"
    md += f"    [View on GitHub]({github_url}){{ .md-button }}\n"

    return md


def main():
    """Generate markdown docs from all examples."""
    # Setup paths
    repo_root = Path(__file__).parent.parent
    examples_dir = repo_root / "examples"
    docs_dir = repo_root / "docs"
    examples_docs_dir = docs_dir / "examples"
    figures_dir = examples_docs_dir / "figures"

    # Create directories
    examples_docs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Find all example Python files
    example_files = sorted(examples_dir.glob("*.py"))

    print("=" * 70)
    print("Generating Example Documentation")
    print("=" * 70)

    for example_file in example_files:
        example_name = example_file.stem
        print(f"\nProcessing: {example_name}")
        print("-" * 70)

        # Run the example
        output = run_example_and_capture(example_file, examples_docs_dir, figures_dir)

        if output is None:
            print(f"⨯ Failed to run {example_name}")
            continue

        # Generate markdown
        md_content = generate_markdown(example_name, output, example_file)

        # Write markdown file
        md_file = examples_docs_dir / f"{example_name}.md"
        md_file.write_text(md_content)

        print(f"✓ Generated {md_file.relative_to(repo_root)}")
        if output["plots"]:
            print(f"  └─ {len(output['plots'])} plot(s) saved to figures/")

    # Create index for examples
    create_examples_index(examples_docs_dir, example_files)

    print("\n" + "=" * 70)
    print("✓ All examples generated successfully!")
    print("=" * 70)


def create_examples_index(examples_docs_dir: Path, example_files: list[Path]) -> None:
    """Create an index page for all examples."""

    md = "# Examples\n\n"
    md += "Comprehensive examples demonstrating stochpw's features and usage patterns.\n\n"

    # Define examples with descriptions
    examples_info = {
        "basic_usage": {
            "title": "Basic Usage",
            "description": (
                "Introduction to fitting permutation weighters " "and assessing balance improvement"
            ),
            "category": "basic",
        },
        "mlp_discriminator": {
            "title": "MLP Discriminator",
            "description": (
                "Using multilayer perceptron discriminators " "for complex confounding patterns"
            ),
            "category": "basic",
        },
        "diagnostics_demo": {
            "title": "Comprehensive Diagnostics",
            "description": (
                "Complete diagnostic workflow with ROC curves, " "calibration, and balance reports"
            ),
            "category": "diagnostics",
        },
        "advanced_features": {
            "title": "Advanced Features",
            "description": "Alternative loss functions, regularization, and early stopping",
            "category": "advanced",
        },
    }

    # Group examples by category
    categories = {
        "basic": {"title": "Basic Usage", "examples": []},
        "diagnostics": {"title": "Diagnostics & Evaluation", "examples": []},
        "advanced": {"title": "Advanced Features", "examples": []},
    }

    for example_file in example_files:
        name = example_file.stem
        if name in examples_info:
            info = examples_info[name]
            categories[info["category"]]["examples"].append((name, info))

    # Generate markdown
    for cat_key in ["basic", "diagnostics", "advanced"]:
        cat = categories[cat_key]
        if cat["examples"]:
            md += f"## {cat['title']}\n\n"
            for name, info in cat["examples"]:
                md += f"### [{info['title']}]({name}.md)\n\n"
                md += f"{info['description']}\n\n"

    index_file = examples_docs_dir / "index.md"
    index_file.write_text(md)
    print(f"✓ Generated examples index at {index_file.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
