"""Generate markdown documentation from examples using jupytext."""

import os
import subprocess
import sys
from pathlib import Path


def run_example_and_capture(
    example_path: Path, output_dir: Path, figures_dir: Path
) -> dict | None:
    """
    Run an example script and capture its output.

    Returns a dict with 'stdout' and any generated plots, or None if failed.
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
            timeout=120,
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


def convert_with_jupytext(example_path: Path, output_md_path: Path) -> bool:
    """
    Convert Python file to markdown using jupytext.

    Args:
        example_path: Path to the .py example file (in percent format)
        output_md_path: Path where the .md file should be written

    Returns:
        True if successful, False otherwise
    """
    try:
        result = subprocess.run(
            ["jupytext", "--to", "md", str(example_path), "--output", str(output_md_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            print(f"Error converting {example_path.name} with jupytext:")
            print(result.stderr)
            return False

        return True

    except Exception as e:
        print(f"Exception during jupytext conversion: {e}")
        return False


def add_github_link_to_markdown(
    md_path: Path, example_path: Path, add_note: bool = False
) -> None:
    """
    Add GitHub link to the markdown file, optionally with a note about execution.

    Args:
        md_path: Path to the markdown file to modify
        example_path: Path to original example file (for GitHub link)
        add_note: If True, add a note about the example not being executed
    """
    # Read the generated markdown
    with open(md_path, "r") as f:
        md_content = f.read()

    additions = []

    # Add note about missing output if requested
    if add_note:
        additions.append("\n## Note\n\n")
        additions.append(
            "!!! info \"Dataset Required\"\n"
            "    This example requires the Lalonde NSW dataset to run. "
            "The code structure is shown above, but output is not available without the dataset.\n\n"
            "    To run this example, you'll need to:\n\n"
            "    1. Download the Lalonde NSW dataset (available from various causal inference repositories)\n"
            "    2. Place it as `background/lalonde_nsw.csv` in the project root\n"
            "    3. Run the example with `python examples/lalonde_experiment.py`\n\n"
        )

    # Add GitHub link at the end
    github_url = f"https://github.com/ddimmery/stochpw/blob/main/examples/{example_path.name}"
    additions.append(f"\n---\n\n[View source on GitHub]({github_url}){{ .md-button }}\n")

    # Append all additions
    modified_content = md_content + "".join(additions)

    # Write back
    with open(md_path, "w") as f:
        f.write(modified_content)


def add_output_and_plots_to_markdown(
    md_path: Path, output: dict, example_path: Path
) -> None:
    """
    Add output and plot sections to the generated markdown.

    Args:
        md_path: Path to the markdown file to modify
        output: Dict with 'stdout' and 'plots' from running the example
        example_path: Path to original example file (for GitHub link)
    """
    # Read the generated markdown
    with open(md_path, "r") as f:
        md_content = f.read()

    # Add output section before the last heading or at the end
    additions = []

    # Add output
    additions.append("\n## Output\n\n")
    additions.append("```\n")
    additions.append(output["stdout"])
    additions.append("```\n\n")

    # Add plots if any
    if output["plots"]:
        additions.append("## Visualizations\n\n")
        for plot_name in output["plots"]:
            # Create a nicer caption from filename
            caption = plot_name.replace("_", " ").replace(".png", "").title()
            additions.append(f"### {caption}\n\n")
            additions.append(f"![{caption}](figures/{plot_name})\n\n")

    # Add GitHub link at the end
    github_url = f"https://github.com/ddimmery/stochpw/blob/main/examples/{example_path.name}"
    additions.append(f"\n---\n\n[View source on GitHub]({github_url}){{ .md-button }}\n")

    # Append all additions
    modified_content = md_content + "".join(additions)

    # Write back
    with open(md_path, "w") as f:
        f.write(modified_content)


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
    print("Generating Example Documentation with Jupytext")
    print("=" * 70)

    for example_file in example_files:
        example_name = example_file.stem
        print(f"\nProcessing: {example_name}")
        print("-" * 70)

        # Convert to markdown with jupytext
        md_file = examples_docs_dir / f"{example_name}.md"
        success = convert_with_jupytext(example_file, md_file)

        if not success:
            print(f"⨯ Failed to convert {example_name} with jupytext")
            continue

        print("✓ Converted to markdown with jupytext")

        # Run the example to get output
        output = run_example_and_capture(example_file, examples_docs_dir, figures_dir)

        if output is None:
            print(f"⚠ Failed to run {example_name}, adding markdown without output")
            # Still add GitHub link with a note about missing execution
            add_github_link_to_markdown(md_file, example_file, add_note=True)
        else:
            # Add output and plots to the markdown
            add_output_and_plots_to_markdown(md_file, output, example_file)

        print(f"✓ Generated {md_file.relative_to(repo_root)}")
        if output and output["plots"]:
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
        "lalonde_experiment": {
            "title": "Lalonde Experiment",
            "description": "Real-world causal inference on the classic Lalonde dataset",
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
