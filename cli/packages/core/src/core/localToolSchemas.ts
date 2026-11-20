/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

// Compact tool descriptions sent to local models. The object below must stay
// valid JSON: the fine-tuning scripts parse this file directly.
export const LOCAL_TOOL_SCHEMAS: Record<
  string,
  { description: string; params: Record<string, string> }
> = {
  "run_shell_command": {
    "description": "Run a bash command (as `bash -c <command>`) in the project folder. Returns combined stdout/stderr, plus the exit code if it failed. Use quiet flags and disable pagers (e.g. `git --no-pager`).",
    "params": {
      "command": "The exact bash command to run.",
      "description": "One short sentence telling the user what the command does.",
      "dir_path": "Optional directory to run in. Must already exist inside the workspace.",
      "is_background": "Set true only for long-running processes such as servers or watchers."
    }
  },
  "read_file": {
    "description": "Read a file and return its contents. Optionally read only a range of lines.",
    "params": {
      "file_path": "Path to the file.",
      "start_line": "Optional 1-based first line to read.",
      "end_line": "Optional 1-based last line to read (inclusive)."
    }
  },
  "write_file": {
    "description": "Write the complete content to a file, creating missing parent folders. Overwrites existing files. Best for new or small files; use replace for targeted edits.",
    "params": {
      "file_path": "Path to the file.",
      "content": "The complete file content. No placeholders."
    }
  },
  "replace": {
    "description": "Replace exact text in an existing file. old_string must match the file exactly, including whitespace and indentation, and must be unique unless allow_multiple is true. Read the file first.",
    "params": {
      "file_path": "Path to the file to edit.",
      "instruction": "One sentence describing the change.",
      "old_string": "The exact existing text to replace, with enough surrounding lines to be unique.",
      "new_string": "The text to put in its place.",
      "allow_multiple": "Set true to replace every occurrence."
    }
  },
  "list_directory": {
    "description": "List the files and subfolders in a directory.",
    "params": {
      "dir_path": "Path of the directory to list."
    }
  },
  "glob": {
    "description": "Find files whose paths match a glob pattern such as `**/*.py` or `src/**/*.ts`. Returns matching paths.",
    "params": {
      "pattern": "The glob pattern.",
      "dir_path": "Optional directory to search in. Defaults to the project root."
    }
  },
  "grep_search": {
    "description": "Search file contents for a regular expression. Returns matching lines with file paths and line numbers.",
    "params": {
      "pattern": "The regular expression to search for.",
      "dir_path": "Optional directory to search in. Defaults to the project root.",
      "include_pattern": "Optional glob limiting which files are searched, e.g. `*.py`."
    }
  },
  "web_fetch": {
    "description": "Fetch a web page over http/https and return its text content.",
    "params": {
      "prompt": "The full URL to fetch, plus what you want to find on the page."
    }
  },
  "google_web_search": {
    "description": "Search the web. Returns titles, URLs and snippets of the top results. Use it for anything current or that you do not know.",
    "params": {
      "query": "The search query."
    }
  }
};
