"""Generate markdown documentation from examples using jupytext and nbconvert."""

import os
import subprocess
from pathlib import Path


def convert_py_to_ipynb(example_path: Path, output_ipynb_path: Path) -> bool:
    """
    Convert Python file to Jupyter notebook using jupytext.

    Args:
        example_path: Path to the .py example file (in percent format)
        output_ipynb_path: Path where the .ipynb file should be written

    Returns:
        True if successful, False otherwise
    """
    try:
        result = subprocess.run(
            ["jupytext", "--to", "ipynb", str(example_path), "--output", str(output_ipynb_path)],
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


def execute_and_convert_notebook(
    ipynb_path: Path, output_md_path: Path, figures_dir: Path
) -> bool:
    """
    Execute notebook and convert to markdown using nbconvert.

    Args:
        ipynb_path: Path to the .ipynb file
        output_md_path: Path where the .md file should be written
        figures_dir: Directory where figures should be saved

    Returns:
        True if successful, False otherwise
    """
    # Change to the examples directory so relative paths work
    original_dir = os.getcwd()
    work_dir = ipynb_path.parent

    try:
        os.chdir(work_dir)

        # Execute and convert the notebook to markdown
        # --execute: Execute the notebook before converting
        # --to markdown: Convert to markdown format
        # --output: Output file path
        # --ExtractOutputPreprocessor.enabled=True: Extract images
        # --NbConvertApp.output_files_dir: Directory for extracted images
        result = subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "markdown",
                "--execute",
                "--output",
                str(output_md_path.absolute()),
                "--ExtractOutputPreprocessor.enabled=True",
                f"--NbConvertApp.output_files_dir={figures_dir.absolute()}",
                str(ipynb_path.name),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            print(f"Error executing/converting {ipynb_path.name}:")
            print(result.stderr)
            return False

        return True

    except subprocess.TimeoutExpired:
        print(f"Timeout executing {ipynb_path.name}")
        return False
    except Exception as e:
        print(f"Exception during nbconvert: {e}")
        return False
    finally:
        os.chdir(original_dir)


def add_github_link_to_markdown(md_path: Path, example_path: Path) -> None:
    """
    Add GitHub link to the markdown file.

    Args:
        md_path: Path to the markdown file to modify
        example_path: Path to original example file (for GitHub link)
    """
    # Read the generated markdown
    with open(md_path, "r") as f:
        md_content = f.read()

    # Add GitHub link at the end
    github_url = f"https://github.com/ddimmery/stochpw/blob/main/examples/{example_path.name}"
    additions = f"\n---\n\n[View source on GitHub]({github_url}){{ .md-button }}\n"

    # Append
    modified_content = md_content + additions

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
    print("Generating Example Documentation with Jupytext + NBConvert")
    print("=" * 70)

    # Track generated notebooks to clean up
    notebooks_to_cleanup = []

    for example_file in example_files:
        example_name = example_file.stem
        print(f"\nProcessing: {example_name}")
        print("-" * 70)

        # Step 1: Convert .py to .ipynb using jupytext
        ipynb_file = examples_dir / f"{example_name}.ipynb"
        success = convert_py_to_ipynb(example_file, ipynb_file)

        if not success:
            print(f"⨯ Failed to convert {example_name} to notebook")
            continue

        print("✓ Converted to notebook with jupytext")
        notebooks_to_cleanup.append(ipynb_file)

        # Step 2: Execute notebook and convert to markdown with nbconvert
        md_file = examples_docs_dir / f"{example_name}.md"
        success = execute_and_convert_notebook(ipynb_file, md_file, figures_dir)

        if not success:
            print(f"⨯ Failed to execute/convert {example_name}")
            continue

        print("✓ Executed and converted to markdown with nbconvert")

        # Step 3: Add GitHub link to markdown
        add_github_link_to_markdown(md_file, example_file)

        print(f"✓ Generated {md_file.relative_to(repo_root)}")

        # Check if figures were generated
        figure_files = list(figures_dir.glob(f"{example_name}_files/*"))
        if figure_files:
            print(f"  └─ {len(figure_files)} plot(s) saved to figures/")

    # Clean up generated notebooks
    print("\nCleaning up temporary notebook files...")
    for notebook in notebooks_to_cleanup:
        if notebook.exists():
            notebook.unlink()
            print(f"  ✓ Removed {notebook.name}")

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
