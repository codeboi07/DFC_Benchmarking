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


MODEL_PROVIDERS = {
    ModelsEnum.GPT_5_1_2025_11_13: "openai",
    ModelsEnum.GPT_4O_2024_08_06: "openai",
    ModelsEnum.GPT_4O_2024_05_13: "openai",
    ModelsEnum.GPT_5_MINI_2025_08_07: "openai",
    ModelsEnum.GPT_4O_MINI_2024_07_18: "openai",
    ModelsEnum.GPT_3_5_TURBO_0125: "openai",
    ModelsEnum.GPT_4_0125_PREVIEW: "openai",
    ModelsEnum.GPT_4_TURBO_2024_04_09: "openai",
}


MODEL_NAMES = {
    "gpt-5.1-2025-11-13": "GPT-5",
    "gpt-4o-2024-08-06": "GPT-4",
    "gpt-4o-2024-05-13": "GPT-4",
    "gpt-5-mini-2025-08-07": "GPT-5",
    "gpt-4o-mini-2024-07-18": "GPT-4",
    "gpt-3.5-turbo-0125": "GPT-3.5",
    "gpt-4-turbo-2024-04-09": "GPT-4",
    "gpt-4-0125-preview": "GPT-4",
}
