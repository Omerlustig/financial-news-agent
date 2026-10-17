import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*doesn't match a supported version.*")
warnings.filterwarnings("ignore", message=".*urllib3.*")

# Ensure news-agent root is in sys.path
news_agent_root = Path(__file__).resolve().parent.parent
if str(news_agent_root) not in sys.path:
    sys.path.insert(0, str(news_agent_root))

import litellm
litellm.suppress_debug_info = True
litellm.drop_params = True

from src.cli import app

if __name__ == "__main__":
    app()
