import os, sys
os.environ["PYTHONWARNINGS"] = "ignore::urllib3.exceptions.NotOpenSSLWarning"
# Suppress transformers/huggingface_hub backend warning
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN")
)


model = ChatHuggingFace(llm=llm)
result = model.invoke("What is RAG?")
print(result.content)

