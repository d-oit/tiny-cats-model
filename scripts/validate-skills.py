#!/usr/bin/env python3
"""Validate agent skills against agentskills.io specification."""

import re
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]

NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
NAME_MAX_LENGTH = 64
DESCRIPTION_MAX_LENGTH = 1024

OPTIONAL_FIELDS: dict = {
    "license": {"type": str, "max_length": 500},
    "compatibility": {"type": str, "max_length": 500},
    "metadata": {"type": dict},
    "allowed-tools": {"type": str},
    "triggers": {"type": list},
}

RED, GREEN, BLUE, NC = "\033[0;31m", "\033[0;32m", "\033[0;34m", "\033[0m"


def log_error(msg: str) -> None:
    print(f"{RED}✗ {msg}{NC}")


def log_success(msg: str) -> None:
    print(f"{GREEN}✓ {msg}{NC}")


def log_info(msg: str) -> None:
    print(f"{BLUE}▶ {msg}{NC}")


def extract_frontmatter(content: str) -> tuple[str | None, str]:
    if not content.startswith("---"):
        return None, content
    end = content.find("\n---", 3)
    if end == -1:
        return None, content
    return content[4:end].strip(), content[end + 4 :].strip()


def validate_name(name: str, dir_name: str) -> list[str]:
    errors = []
    if not name or len(name) < 1:
        errors.append("Name is empty")
    elif len(name) > NAME_MAX_LENGTH:
        errors.append(f"Name too long: {len(name)} chars (max {NAME_MAX_LENGTH})")
    if not NAME_PATTERN.match(name):
        errors.append(f"Name '{name}' has invalid characters")
    if name != dir_name:
        errors.append(f"Name '{name}' != directory '{dir_name}'")
    return errors


def validate_description(desc: str) -> list[str]:
    errors = []
    if not desc or not desc.strip():
        errors.append("Description is empty")
    elif len(desc) > DESCRIPTION_MAX_LENGTH:
        errors.append(f"Description too long: {len(desc)} chars")
    return errors


def validate_optional_fields(fm: dict) -> list[str]:
    errors = []
    for field, rules in OPTIONAL_FIELDS.items():
        if field not in fm:
            continue
        val = fm[field]
        expected_type = rules["type"]
        if not isinstance(val, expected_type):
            errors.append(f"Field '{field}' wrong type")
            continue
        if expected_type is str and "max_length" in rules:
            if len(val) > rules["max_length"]:
                errors.append(f"Field '{field}' too long")
        if field == "metadata" and isinstance(val, dict):
            for k, v in val.items():
                if not isinstance(k, str) or not isinstance(v, str):
                    errors.append(f"Metadata '{k}' must be string")
        if field == "triggers" and isinstance(val, list):
            for i, t in enumerate(val):
                if not isinstance(t, str):
                    errors.append(f"Trigger {i} must be string")
    return errors


def validate_skill(skill_path: Path) -> list[str]:
    errors = []
    skill_file = skill_path / "SKILL.md"
    if not skill_file.exists():
        errors.append(f"Missing SKILL.md in {skill_path}")
        return errors
    try:
        content = skill_file.read_text(encoding="utf-8")
    except Exception as e:
        errors.append(f"Cannot read SKILL.md: {e}")
        return errors
    fm_yaml, _ = extract_frontmatter(content)
    if fm_yaml is None:
        errors.append("No YAML frontmatter")
        return errors
    try:
        fm = yaml.safe_load(fm_yaml)
    except yaml.YAMLError as e:
        errors.append(f"Invalid YAML: {e}")
        return errors
    if not isinstance(fm, dict):
        errors.append("Frontmatter must be a mapping")
        return errors
    if "name" not in fm:
        errors.append("Missing 'name'")
    else:
        errors.extend(validate_name(str(fm["name"]), skill_path.name))
    if "description" not in fm:
        errors.append("Missing 'description'")
    else:
        errors.extend(validate_description(str(fm["description"])))
    errors.extend(validate_optional_fields(fm))
    return errors


def validate_skills_directory(skills_dir: Path) -> tuple[int, int]:
    valid, invalid = 0, 0
    skill_dirs = sorted(
        [d for d in skills_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    log_info(f"Validating {len(skill_dirs)} skill(s)...")
    print()
    for sd in skill_dirs:
        errs = validate_skill(sd)
        if errs:
            invalid += 1
            log_error(f"Skill '{sd.name}' has {len(errs)} error(s):")
            for e in errs:
                print(f"    • {e}")
            print()
        else:
            valid += 1
            log_success(f"Skill '{sd.name}' is valid")
    return valid, invalid


def main() -> int:
    skills_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent.parent / ".agents" / "skills"
    )
    skills_dir = skills_path.resolve()
    print()
    print("=" * 59)
    print("  Agent Skills Validation (agentskills.io specification)")
    print("=" * 59)
    print()
    valid, invalid = validate_skills_directory(skills_dir)
    print()
    print("-" * 59)
    print()
    if invalid:
        log_error(f"Validation failed: {invalid} skill(s) with errors")
        print()
        print("See: https://agentskills.io/specification#validation")
        return 1
    log_success(f"All {valid} skills passed validation!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
