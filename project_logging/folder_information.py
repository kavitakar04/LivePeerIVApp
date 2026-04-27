import ast
import datetime
from pathlib import Path
from typing import List, Dict, Tuple


class FolderAnalyzer:
    """Analyzes project folder structure and extracts functions/classes from Python files."""

    def __init__(self, root_path: str = None):
        if root_path is None:
            # Get the project root (parent of logging folder)
            self.root_path = Path(__file__).parent.parent
        else:
            self.root_path = Path(root_path)

        self.logging_history_path = Path(__file__).parent / "logging_history"

    def extract_functions_and_classes(self, file_path: str) -> Dict[str, List[str]]:
        """Extract function and class names from a Python file."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            tree = ast.parse(content)

            functions = []
            classes = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)

            # Also get function I/O details
            io = self.extract_function_io(file_path)

            return {
                "functions": functions,
                "classes": classes,
                "function_io": io,  # Now included in output
            }

        except Exception as e:
            return {"functions": [f"Error parsing file: {str(e)}"], "classes": [], "function_io": {}}

    def extract_function_io(self, file_path: str) -> Dict[str, Dict[str, List[str]]]:
        """Extract function input and output parameters."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            tree = ast.parse(content)
            function_io = {}

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_name = node.name
                    function_io[function_name] = {
                        "inputs": [],
                        "outputs": [],  # Outputs cannot be statically inferred reliably without more analysis
                    }

                    # Regular arguments
                    for arg in node.args.args:
                        function_io[function_name]["inputs"].append(arg.arg)

                    # Keyword-only arguments
                    for arg in node.args.kwonlyargs:
                        function_io[function_name]["inputs"].append(arg.arg)

                    # *args
                    if node.args.vararg:
                        function_io[function_name]["inputs"].append(f"*{node.args.vararg.arg}")

                    # **kwargs
                    if node.args.kwarg:
                        function_io[function_name]["inputs"].append(f"**{node.args.kwarg.arg}")

            return function_io

        except Exception as e:
            return {"error": f"Error extracting function IO: {str(e)}"}

    def get_folder_structure(self) -> List[Tuple[str, int]]:
        """Get the folder structure with indentation levels."""
        structure = []

        # Directories to skip entirely
        skip_dirs = {
            "venv",
            ".venv",
            "env",
            ".env",
            "node_modules",
            "__pycache__",
            ".git",
            ".pytest_cache",
            "dist",
            "build",
        }

        def walk_directory(path: Path, level: int = 0):
            # Skip if this is a directory we want to ignore
            if path.name in skip_dirs:
                return

            try:
                items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))

                for item in items:
                    # Skip hidden files, common ignore patterns, and virtual environments
                    if (
                        item.name.startswith(".")
                        or item.name in skip_dirs
                        or item.name.endswith(".pyc")
                        or item.name.endswith(".pyo")
                    ):
                        continue

                    structure.append((str(item.relative_to(self.root_path)), level))

                    if item.is_dir():
                        walk_directory(item, level + 1)

            except PermissionError:
                structure.append((f"Permission denied: {path}", level))
            except Exception as e:
                structure.append((f"Error accessing: {path} - {str(e)}", level))

        walk_directory(self.root_path)
        return structure

    def generate_report(self) -> str:
        """Generate a comprehensive folder and code analysis report."""
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PROJECT STRUCTURE AND CODE ANALYSIS REPORT")
        report_lines.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Root Path: {self.root_path}")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Section 1: Folder Structure
        report_lines.append("1. FOLDER STRUCTURE")
        report_lines.append("-" * 40)

        structure = self.get_folder_structure()
        for path, level in structure:
            indent = "  " * level
            if Path(self.root_path / path).is_dir():
                report_lines.append(f"{indent}📁 {Path(path).name}/")
            else:
                report_lines.append(f"{indent}📄 {Path(path).name}")

        report_lines.append("")

        # Section 2: Python Files Analysis
        report_lines.append("2. PYTHON FILES ANALYSIS")
        report_lines.append("-" * 40)

        python_files = []
        for path, _ in structure:
            full_path = self.root_path / path
            if full_path.is_file() and path.endswith(".py"):
                python_files.append(str(full_path))

        if not python_files:
            report_lines.append("No Python files found.")
        else:
            for file_path in python_files:
                relative_path = Path(file_path).relative_to(self.root_path)
                report_lines.append(f"\nFile: {relative_path}")
                report_lines.append("-" * len(f"File: {relative_path}"))

                analysis = self.extract_functions_and_classes(file_path)

                if analysis["classes"]:
                    report_lines.append("Classes:")
                    for class_name in analysis["classes"]:
                        report_lines.append(f"  • {class_name}")

                if analysis["functions"]:
                    report_lines.append("Functions:")
                    for func_name in analysis["functions"]:
                        report_lines.append(f"  • {func_name}")

                if not analysis["classes"] and not analysis["functions"]:
                    report_lines.append("  No classes or functions found.")

        # Section 3: Summary Statistics
        report_lines.append("")
        report_lines.append("3. SUMMARY STATISTICS")
        report_lines.append("-" * 40)

        total_files = len([p for p, _ in structure if Path(self.root_path / p).is_file()])
        total_dirs = len([p for p, _ in structure if Path(self.root_path / p).is_dir()])
        total_py_files = len(python_files)

        total_functions = 0
        total_classes = 0

        for file_path in python_files:
            analysis = self.extract_functions_and_classes(file_path)
            total_functions += len(analysis["functions"])
            total_classes += len(analysis["classes"])

        report_lines.append(f"Total Directories: {total_dirs}")
        report_lines.append(f"Total Files: {total_files}")
        report_lines.append(f"Python Files: {total_py_files}")
        report_lines.append(f"Total Classes: {total_classes}")
        report_lines.append(f"Total Functions: {total_functions}")

        # Section 4: Dependencies and Data Handling Checkpoints
        report_lines.append("")
        report_lines.append("4. POTENTIAL DATA HANDLING CHECKPOINTS")
        report_lines.append("-" * 40)

        data_keywords = ["load", "save", "read", "write", "export", "import", "process", "filter", "transform"]

        for file_path in python_files:
            relative_path = Path(file_path).relative_to(self.root_path)
            analysis = self.extract_functions_and_classes(file_path)

            data_functions = []
            for func in analysis["functions"]:
                if any(keyword in func.lower() for keyword in data_keywords):
                    data_functions.append(func)

            if data_functions:
                report_lines.append(f"\n{relative_path}:")
                for func in data_functions:
                    report_lines.append(f"  • {func}")

        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("End of Report")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def save_report(self) -> str:
        """Generate and save the report to a timestamped file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"folder_analysis_{timestamp}.txt"
        filepath = self.logging_history_path / filename

        report_content = self.generate_report()

        with open(filepath, "w", encoding="utf-8") as file:
            file.write(report_content)

        return str(filepath)


def run_analysis():
    """Main function to run the folder analysis."""
    analyzer = FolderAnalyzer()
    report_path = analyzer.save_report()

    print("✅ Folder analysis complete!")
    print(f"📄 Report saved to: {report_path}")

    # Also print a summary to console
    print("\n" + "=" * 50)
    print("QUICK SUMMARY")
    print("=" * 50)

    structure = analyzer.get_folder_structure()

    python_files = [p for p, _ in structure if Path(analyzer.root_path / p).suffix == ".py"]

    print(f"📁 Total directories: {len([p for p, _ in structure if Path(analyzer.root_path / p).is_dir()])}")
    print(f"📄 Total files: {len([p for p, _ in structure if Path(analyzer.root_path / p).is_file()])}")
    print(f"🐍 Python files: {len(python_files)}")

    return report_path


if __name__ == "__main__":
    run_analysis()
