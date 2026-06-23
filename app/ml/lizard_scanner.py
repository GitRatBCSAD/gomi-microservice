import os
import lizard 

SOURCE_EXTENSIONS = {
        ".py", ".js", ".ts", ".java", ".tsx", ".go", ".c", ".cpp", ".cs"
        }
SKIP_DIRS = {
        ".git", "node_modules", "vendor", "__pycache__", "build", "dist", ".idea", ".vscode", ".gradle", ".mvn", ".settings", ".cache"
        }

def get_source_files(repo_path: str) -> list[str]:
    """
    Finds all valid source code files in the repo
    """

    source_files = []
    for root, dirs, filenames in os.walk(repo_path):
        # Skip directories in SKIP_DIRS
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]

        for filename in filenames:
            if any(filename.endswith(ext) for ext in SOURCE_EXTENSIONS):
                source_files.append(os.path.join(root, filename))

    return source_files

def compute_complexity(filepath: str) -> dict:
    """
    Scans a source file using lizard and computes cyclomatic complexity metrics.
    Returns a dictionary of averages across all functions in the file.
    """
    try: 
        result = lizard.analyze_file(filepath)
    except Exception as e:
        return {
                "AvgCCN": 0.0,
                "AvgNLOC": 0.0,
                "function_cnt": 0,
                "PARAM": 0.0
                }

    if not result.function_list:
        return {
                "AvgCCN": 0.0,
                "AvgNLOC": 0.0,
                "function_cnt": 0,
                "PARAM": 0.0
                }
    functions = result.function_list
    num_funcs = len(functions)

    avg_ccn = sum(func.cyclomatic_complexity for func in functions) / num_funcs
    avg_nloc = sum(func.nloc for func in functions) / num_funcs
    avg_param = sum(func.parameter_count for func in functions) / num_funcs

    return {
            "AvgCCN": avg_ccn,
            "AvgNLOC": avg_nloc,
            "function_cnt": num_funcs,
            "PARAM": avg_param
            }

