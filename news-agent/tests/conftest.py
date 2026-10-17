import sys
from pathlib import Path

# Ensure news-agent root is in sys.path
news_agent_root = Path(__file__).resolve().parent.parent
if str(news_agent_root) not in sys.path:
    sys.path.insert(0, str(news_agent_root))
