from src.skills.registry import SkillRegistry

def build_skill_schemas(registry: SkillRegistry):
    return registry.get_all_schemas()
