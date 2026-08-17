"""Typed-confirm gate (plan_and_execute.py pattern): exact 'yes' or abort."""


def typed_yes(prompt):
    try:
        answer = input(prompt)
    except EOFError:
        answer = ""
    return answer.strip() == "yes"
