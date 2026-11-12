# Author: Sunny Zhou
# Email: shadowkingcary@gmail.com

"""Execute every zero-argument test function, including files without a main block."""
from project_paths import RESULTS, FIGURES, PROTOCOL, REFERENCE, WORK, artifact_label, external
import importlib.util
import inspect
from pathlib import Path


def main():
    count = 0
    for path in sorted((Path(__file__).resolve().parents[1] / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_") and function.__module__ == module.__name__:
                if inspect.signature(function).parameters:
                    raise ValueError(f"{path.name}:{name} requires a fixture-aware runner")
                function()
                count += 1
        print(f"PASS {path.name}")
    print(f"PASS {count} test functions")


if __name__ == "__main__":
    main()
