import os
from dotenv import load_dotenv

load_dotenv()

furthr = {
    'host': "https://furthr.informatik.uni-marburg.de/",
    'api_key': os.getenv("FURTHRMIND_API_KEY"),
    'project_id': "63fc6e834b788da18c9ecc2c",
    'model_group_id': "6761371c22ff6c2edcaf731a"
}