from agentdojo.strenum import StrEnum


class ModelsEnum(StrEnum):
    """Currently supported models."""
    GPT_5_1_2025_11_13 = "gpt-5.1-2025-11-13"
    """GPT-5.1"""
    GPT_4O_2024_08_06 = "gpt-4o-2024-08-06"
    """GPT-4o"""
    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"
    """GPT-4o"""
    GPT_5_MINI_2025_08_07 = "gpt-5-mini-2025-08-07"
    """GPT-5 Mini"""
    GPT_4O_MINI_2024_07_18 = "gpt-4o-mini-2024-07-18"
    """GPT-4o Mini"""
    GPT_4_0125_PREVIEW = "gpt-4-0125-preview"
    """GPT-4 Turbo Preview 0125"""
    GPT_3_5_TURBO_0125 = "gpt-3.5-turbo-0125"
    """GPT-3.5 Turbo 0125"""
    GPT_4_TURBO_2024_04_09 = "gpt-4-turbo-2024-04-09"
    """GPT-4 Turbo"""
    # AWS Bedrock-hosted Claude (inference-profile IDs). Provider "bedrock".
    BEDROCK_CLAUDE_SONNET_4_5 = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    """Claude Sonnet 4.5 (Bedrock)"""
    BEDROCK_CLAUDE_HAIKU_4_5 = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    """Claude Haiku 4.5 (Bedrock)"""
    BEDROCK_CLAUDE_SONNET_4_6 = "us.anthropic.claude-sonnet-4-6"
    """Claude Sonnet 4.6 (Bedrock)"""
    BEDROCK_CLAUDE_OPUS_4_8 = "us.anthropic.claude-opus-4-8"
    """Claude Opus 4.8 (Bedrock)"""
    LITELLM_QWEN3_235B = "qwen3-235b"
    """Qwen3-235B via LiteLLM (Bedrock Converse application inference profile, us-west-2)."""
    LITELLM_DEEPSEEK_V3_2 = "deepseek-v3.2"
    """DeepSeek V3.2 via LiteLLM (Bedrock Converse application inference profile, us-west-2)."""
    LITELLM_GPT_OSS_120B = "gpt-oss-120b"
    """OpenAI GPT-OSS 120B via LiteLLM (Bedrock Converse application inference profile, us-west-2)."""
    LITELLM_KIMI_K2_5 = "kimi-k2.5"
    """Moonshot AI Kimi K2.5 via LiteLLM (Bedrock Converse application inference profile, us-west-2)."""
    LITELLM_MINIMAX_M2_5 = "minimax-m2.5"
    """MiniMax M2.5 via LiteLLM (Bedrock Converse application inference profile, us-west-2)."""


MODEL_PROVIDERS = {
    ModelsEnum.GPT_5_1_2025_11_13: "openai",
    ModelsEnum.GPT_4O_2024_08_06: "openai",
    ModelsEnum.GPT_4O_2024_05_13: "openai",
    ModelsEnum.GPT_5_MINI_2025_08_07: "openai",
    ModelsEnum.GPT_4O_MINI_2024_07_18: "openai",
    ModelsEnum.GPT_3_5_TURBO_0125: "openai",
    ModelsEnum.GPT_4_0125_PREVIEW: "openai",
    ModelsEnum.GPT_4_TURBO_2024_04_09: "openai",
    ModelsEnum.BEDROCK_CLAUDE_SONNET_4_5: "bedrock",
    ModelsEnum.BEDROCK_CLAUDE_HAIKU_4_5: "bedrock",
    ModelsEnum.BEDROCK_CLAUDE_SONNET_4_6: "bedrock",
    ModelsEnum.BEDROCK_CLAUDE_OPUS_4_8: "bedrock",
    ModelsEnum.LITELLM_QWEN3_235B: "litellm",
    ModelsEnum.LITELLM_DEEPSEEK_V3_2: "litellm",
    ModelsEnum.LITELLM_GPT_OSS_120B: "litellm",
    ModelsEnum.LITELLM_KIMI_K2_5: "litellm",
    ModelsEnum.LITELLM_MINIMAX_M2_5: "litellm",
}


MODEL_NAMES = {
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": "Claude",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": "Claude",
    "us.anthropic.claude-sonnet-4-6": "Claude",
    "us.anthropic.claude-opus-4-8": "Claude",
    "qwen3-235b": "Qwen",
    "deepseek-v3.2": "DeepSeek",
    "gpt-oss-120b": "GPT-OSS",
    "kimi-k2.5": "Kimi",
    "minimax-m2.5": "MiniMax",
    "gpt-5.1-2025-11-13": "GPT-5",
    "gpt-4o-2024-08-06": "GPT-4",
    "gpt-4o-2024-05-13": "GPT-4",
    "gpt-5-mini-2025-08-07": "GPT-5",
    "gpt-4o-mini-2024-07-18": "GPT-4",
    "gpt-3.5-turbo-0125": "GPT-3.5",
    "gpt-4-turbo-2024-04-09": "GPT-4",
    "gpt-4-0125-preview": "GPT-4",
}
