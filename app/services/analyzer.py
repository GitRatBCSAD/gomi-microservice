import tempfile
import subprocess
from typing import List, Optional
from ..schemas.analyze import FileRiskResult, SHAPBreakdown

class AnalyzerService:
    @staticmethod
    def analyze_repository(repo_url: str, branch: str, access_token: Optional[str] = None) -> List[FileRiskResult]:
       # Create a temporary directory to clone the repository 
        with tempfile.TemporaryDirectory() as temp_dir:
            clone_url = repo_url
            if access_token:
                clone_url = clone_url.replace("https://", f"https://x-access-token:{access_token}@")
            try:
                print(f"Cloning repository into {temp_dir}...")
                subprocess.run(
                    ["git", "clone", "--shallow-since=6 months ago", "--single-branch", "--branch", branch, clone_url, temp_dir], 
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError as e:
                raise ValueError(f"Git Clone Failed: {e.stderr.decode()}")

            # Handle our gomi model logic here 

            return [
                FileRiskResult(
                    filename="src/auth/login.js",
                    risk_score=0.85,
                    sentiment_score=0.9,
                    complexity_score=0.8,
                    low_confidence=False,
                    shap_breakdown=SHAPBreakdown(
                        base_rate=0.4,
                        sentiment_contrib=0.2,
                        complexity_contrib=0.15,
                        low_info_contrib=0.1,
                        entropy_contrib=0.0,
                        ndev_contrib=0.0,
                        age_contrib=0.0,
                        commits_contrib=0.0
                    )
                )
            ]
