# ML Project - Antigravity Rules

## Project Overview
This project performs exploratory data analysis and builds an XGBoost model using scikit-learn datasets.

## Coding Standards

### Language and Tools
- Use Python 3.11+
- Use `uv` for package management (never pip)
- Use `polars` for data manipulation (not pandas)
- Use `ruff` for linting and formatting
- Use `pytest` for testing

### Code Style
- Use type annotations for all function parameters (one parameter per line)
- All private functions must start with underscore (`_`) and be placed at the top of the file
- Public functions follow after private functions
- Functions should be no more than 30-50 lines
- Two blank lines between function definitions
- Use multi-line imports

### Logging
Always use this logging configuration:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
```

### Constants
- Do not hard-code constants inside functions
- Declare constants at the top of the file with type annotations

### After Writing Python Files
- Always run `uv run ruff check --fix <filename>` after writing Python files
- Always run `uv run python -m py_compile <filename>` to verify syntax

### Output
- Save plots to the `output/` directory
- Use `logging.info()` for progress messages
- Pretty-print dictionaries in log messages using `json.dumps(data, indent=2, default=str)`
