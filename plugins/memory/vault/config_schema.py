"""Vault provider's declared config surface — rendered by the generic desktop panel."""

from plugins.memory.config_schema import (
    KIND_TEXT,
    ProviderConfigSchema,
    ProviderField,
)

CONFIG_SCHEMA = ProviderConfigSchema(
    name="vault",
    label="Obsidian Vault",
    fields=(
        ProviderField(
            key="vault_path",
            label="Vault path",
            kind=KIND_TEXT,
            default="",
            description=(
                "Absolute path to the Obsidian vault root "
                "(Projects/, Concepts/, References/, Decisions/, People/). "
                "Leave empty to use the documented default. The "
                "OBSIDIAN_VAULT_PATH environment variable overrides this."
            ),
            placeholder=r"C:\Users\Jie\HermesMemory",
            inline=True,
        ),
    ),
)
