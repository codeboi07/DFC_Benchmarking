from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkTaskContext(BaseModel):
    benchmark_name: str
    benchmark_version: str | None = None
    suite_name: str | None = None
    task_id: str | None = None
    task_kind: Literal["user", "injection", "unknown"] = "unknown"
    preamble: str


class PreambleFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(description="Stable snake_case identifier for the extracted fact.")
    value: str = Field(description="Exact value or phrase from the preamble.")


class PreambleExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[PreambleFact] = Field(
        min_length=1,
        description="Extracted preamble facts as explicit key/value pairs.",
    )

    @classmethod
    def from_dict(cls, facts: dict[str, str]) -> PreambleExtraction:
        return cls(facts=[PreambleFact(key=key, value=value) for key, value in facts.items()])


class GeneratedPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(description="Stable identifier for this policy, such as send_email_recipient.")
    pgn: str = Field(
        description=(
            "Complete executable PGN policy text as newline-separated clauses. "
            "Must include every required line: SINK (and SOURCE/SOURCE REQUIRED when needed), "
            "DIMENSION when grounding to preamble or tool output, CONSTRAINT, ON FAIL KILL, "
            "and DESCRIPTION. Do not put only the SINK line here."
        ),
    )
    description: str = Field(
        description="Short human-readable summary that should match the DESCRIPTION clause inside pgn.",
    )
    applies_to_relation: str = Field(
        description="Primary sink relation this policy governs, such as SendEmailInput.",
    )
    applies_to_event: str | None = Field(
        default=None,
        description="Optional event label such as tool_call:send_email.",
    )
    rationale: str = Field(description="Why this policy is needed for the task.")


class GeneratedPolicySet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policies: list[GeneratedPolicy]


class PolicyRepairDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delete: bool = Field(
        description=(
            "Set true only when the policy cannot be repaired to a valid schema-grounded PGN. "
            "Do not set true because the trusted task does not use the sink's tool; "
            "pre-emptive external-sink guards should be repaired, not deleted."
        ),
    )
    repaired_pgn: str | None = Field(
        default=None,
        description=(
            "Complete repaired PGN with SINK, DIMENSION/SOURCE as needed, CONSTRAINT, "
            "ON FAIL KILL, and DESCRIPTION lines. Required when delete is false."
        ),
    )
    repaired_description: str | None = None
    rationale: str = Field(
        description="Explain the repair or, if deleting, why the policy is unrecoverable (not merely unrelated to the task).",
    )


class DFCViolation(BaseModel):
    event_type: str
    relation: str
    attempted_payload: dict[str, Any]
    policy_ids: list[str] = Field(default_factory=list)
    policy_descriptions: list[str]
    raw_error: str | None = None


class ViolationIdentification(BaseModel):
    policy_ids: list[str] = Field(default_factory=list)
    policy_descriptions: list[str] = Field(default_factory=list)


class RelationSchema(BaseModel):
    name: str
    columns: dict[str, str]
    column_descriptions: dict[str, str] = Field(default_factory=dict)
    description: str = ""
    tool_name: str | None = None
    source_key_by_column: dict[str, str] = Field(default_factory=dict)


class RuntimeSchema(BaseModel):
    preamble_relation: str = "PreambleData"
    tool_input_relations: list[RelationSchema] = Field(default_factory=list)
    tool_output_relations: list[RelationSchema] = Field(default_factory=list)
    prompt_input_relation: RelationSchema | None = None
    assistant_response_relation: RelationSchema | None = None

    @classmethod
    def from_tools(
        cls,
        functions: dict[str, Any],
        *,
        functions_runtime: Any | None = None,
        env: Any | None = None,
    ) -> RuntimeSchema:
        from dfc_agent_framework_integration.events import (
            assistant_response_relation,
            column_specs_from_function,
            column_specs_from_return_type,
            input_table_name,
            output_table_name,
            prompt_input_relation,
        )

        tool_inputs: list[RelationSchema] = []
        tool_outputs: list[RelationSchema] = []
        for name, function in functions.items():
            input_cols, input_descriptions = column_specs_from_function(function)
            tool_inputs.append(
                RelationSchema(
                    name=input_table_name(name),
                    columns=input_cols,
                    column_descriptions=input_descriptions,
                    description=function.description,
                    tool_name=name,
                )
            )
            output_cols, output_descriptions, source_key_by_column = column_specs_from_return_type(
                function,
                functions_runtime=functions_runtime,
                env=env,
            )
            tool_outputs.append(
                RelationSchema(
                    name=output_table_name(name),
                    columns=output_cols,
                    column_descriptions=output_descriptions,
                    description=function.description,
                    tool_name=name,
                    source_key_by_column=source_key_by_column,
                )
            )
        return cls(
            tool_input_relations=tool_inputs,
            tool_output_relations=tool_outputs,
            prompt_input_relation=prompt_input_relation(),
            assistant_response_relation=assistant_response_relation(),
        )

    def all_relations(self) -> list[RelationSchema]:
        relations = list(self.tool_input_relations) + list(self.tool_output_relations)
        if self.prompt_input_relation is not None:
            relations.append(self.prompt_input_relation)
        if self.assistant_response_relation is not None:
            relations.append(self.assistant_response_relation)
        return relations

    def relation_names(self) -> set[str]:
        names = {relation.name for relation in self.all_relations()}
        names.add(self.preamble_relation)
        return names


class DeletedPolicyRecord(BaseModel):
    policy_id: str
    rationale: str
    error: str | None = None


class PolicyRegistrationRecord(BaseModel):
    policy_id: str
    repair_attempts: int
    outcome: Literal["registered", "deleted"]


class DFCTaskDiagnostics(BaseModel):
    extracted_facts: dict[str, str] = Field(default_factory=dict)
    generated_policies: list[GeneratedPolicy] = Field(default_factory=list)
    registered_policy_ids: list[str] = Field(default_factory=list)
    deleted_policies: list[DeletedPolicyRecord] = Field(default_factory=list)
    policy_registration: list[PolicyRegistrationRecord] = Field(default_factory=list)
    policy_fire_counts: dict[str, int] = Field(default_factory=dict)
    validation_events: list[dict[str, Any]] = Field(default_factory=list)
