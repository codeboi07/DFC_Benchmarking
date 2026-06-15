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
    BEDROCK_CLAUDE_SONNET_4_6 = "arn:aws:bedrock:us-east-1:920736616554:application-inference-profile/zkck5e7zn6gf"
    """Claude Sonnet 4.6 (Bedrock application inference profile)"""
    BEDROCK_CLAUDE_HAIKU_4_5_APP = "arn:aws:bedrock:us-east-1:920736616554:application-inference-profile/fpcgtipi5lpq"
    """Claude Haiku 4.5 (Bedrock application inference profile, tagged aup2005)"""
    BEDROCK_LLAMA_3_1_70B = "llama-3-1-70b"
    """Llama 3.1 70B Instruct (Bedrock Converse). ARN resolved in BEDROCK_CONVERSE_ARNS."""
    BEDROCK_GPT_OSS_120B = "gpt-oss-120b"
    """OpenAI GPT-OSS 120B (Bedrock Converse). ARN resolved in BEDROCK_CONVERSE_ARNS."""


# Clean model names -> Bedrock application-inference-profile ARNs (tagged aup2005), used by the
# `bedrock_converse` provider. Keeping the enum value clean gives readable run dirs (runs/gpt-oss-120b/).
BEDROCK_CONVERSE_ARNS = {
    "llama-3-1-70b": "arn:aws:bedrock:us-east-1:920736616554:application-inference-profile/e3ha78l6k6eq",
    "gpt-oss-120b": "arn:aws:bedrock:us-east-1:920736616554:application-inference-profile/5qjesloag0zm",
}


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
    ModelsEnum.BEDROCK_CLAUDE_HAIKU_4_5_APP: "bedrock",
    ModelsEnum.BEDROCK_LLAMA_3_1_70B: "bedrock_converse",
    ModelsEnum.BEDROCK_GPT_OSS_120B: "bedrock_converse",
}


MODEL_NAMES = {
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": "Claude",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": "Claude",
    "arn:aws:bedrock:us-east-1:920736616554:application-inference-profile/zkck5e7zn6gf": "Claude",
    "arn:aws:bedrock:us-east-1:920736616554:application-inference-profile/fpcgtipi5lpq": "Claude",
    "llama-3-1-70b": "AI assistant",
    "gpt-oss-120b": "AI assistant",
    "gpt-5.1-2025-11-13": "GPT-5",
    "gpt-4o-2024-08-06": "GPT-4",
    "gpt-4o-2024-05-13": "GPT-4",
    "gpt-5-mini-2025-08-07": "GPT-5",
    "gpt-4o-mini-2024-07-18": "GPT-4",
    "gpt-3.5-turbo-0125": "GPT-3.5",
    "gpt-4-turbo-2024-04-09": "GPT-4",
    "gpt-4-0125-preview": "GPT-4",
}
