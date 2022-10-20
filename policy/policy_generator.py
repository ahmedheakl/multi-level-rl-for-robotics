from linear_policy import Linear

def none_policy():
    return None

policy_generator = dict()
policy_generator["linear"] = Linear
policy_generator["none"] = none_policy

"""
To Do:
- Add human policy
- Add other desired policies
"""