The PostToolUse Hook for command: 
Tells the model to run the ruff and py_compile files, it runs after every file write/edit that the model does.

PostToolUse prompt: 
Gives the prompt/instruction to the model, to check if a corresponding test file exists in a tests/ directory. 
If no test file exists, create one with pytest covering the public functions.

PreToolUse:

Runs a blocking script before any Bash command that prevents force pushes.
