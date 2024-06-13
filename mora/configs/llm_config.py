
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import yaml
from pydantic import BaseModel
from pydantic import field_validator


class YamlModel(BaseModel):
    """Base class for yaml model"""

    extra_fields: Optional[Dict[str, str]] = None

    @classmethod
    def read_yaml(cls, file_path: Path, encoding: str = "utf-8") -> Dict:
        """Read yaml file and return a dict"""
        if not file_path.exists():
            return {}
        with open(file_path, "r", encoding=encoding) as file:
            return yaml.safe_load(file)

    @classmethod
    def from_yaml_file(cls, file_path: Path) -> "YamlModel":
        """Read yaml file and return a YamlModel instance"""
        return cls(**cls.read_yaml(file_path))

    def to_yaml_file(self, file_path: Path, encoding: str = "utf-8") -> None:
        """Dump YamlModel instance to yaml file"""
        with open(file_path, "w", encoding=encoding) as file:
            yaml.dump(self.model_dump(), file)


class LLMType(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CLAUDE = "claude"  # alias name of anthropic
    SPARK = "spark"
    ZHIPUAI = "zhipuai"
    FIREWORKS = "fireworks"
    OPEN_LLM = "open_llm"
    GEMINI = "gemini"
    METAGPT = "metagpt"
    AZURE = "azure"
    OLLAMA = "ollama"
    QIANFAN = "qianfan"  # Baidu BCE
    DASHSCOPE = "dashscope"  # Aliyun LingJi DashScope
    MOONSHOT = "moonshot"
    MISTRAL = "mistral"
    YI = "yi"  # lingyiwanwu

    def __missing__(self, key):
        return self.OPENAI


class LLMConfig(YamlModel):
    """Config for LLM

    OpenAI: https://github.com/openai/openai-python/blob/main/src/openai/resources/chat/completions.py#L681
    Optional Fields in pydantic: https://docs.pydantic.dev/latest/migration/#required-optional-and-nullable-fields
    """

    api_key: str = "sk-"
    api_type: LLMType = LLMType.OPENAI
    base_url: str = "https://api.openai.com/v1"
    api_version: Optional[str] = None
    azure_endpoint: str=None
    model: Optional[str] = None  # also stands for DEPLOYMENT_NAME
    pricing_plan: Optional[str] = None  # Cost Settlement Plan Parameters.

    # For Cloud Service Provider like Baidu/ Alibaba
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    endpoint: Optional[str] = None  # for self-deployed model on the cloud

    # For Spark(Xunfei), maybe remove later
    app_id: Optional[str] = None
    api_secret: Optional[str] = None
    domain: Optional[str] = None

    # For Chat Completion
    max_token: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop: Optional[str] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: Optional[int] = None
    n: Optional[int] = None
    stream: bool = False
    logprobs: Optional[bool] = None  # https://cookbook.openai.com/examples/using_logprobs
    top_logprobs: Optional[int] = None
    timeout: int = 600

    # For Network
    proxy: Optional[str] = None

    # Cost Control
    calc_usage: bool = True

    @field_validator("api_key")
    @classmethod
    def check_llm_key(cls, v):
        if v in ["", None, "YOUR_API_KEY"]:
            raise ValueError("Please set your API key in config2.yaml")
        return v

    @field_validator("timeout")
    @classmethod
    def check_timeout(cls, v):
        return v or LLM_API_TIMEOUT
