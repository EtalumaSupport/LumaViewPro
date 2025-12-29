from pydantic import BaseModel

def model_param_description(description:str, model: type[BaseModel]) -> str:
    lines = []
    for name, field in model.model_fields.items():
        if field.description:
            lines.append(f"- **{name}**: {field.description}")
        else:
            lines.append(f"- **{name}**")
    return description + "\n\n### Parameters\n" + "\n".join(lines)