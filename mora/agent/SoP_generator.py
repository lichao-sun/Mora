
from mora.agent import Role

class ProjectManager(Role):
    """
    Represents a Project Manager role responsible for overseeing project execution and team efficiency.

    Attributes:
        name (str): Name of the project manager.
        profile (str): Role profile, default is 'Project Manager'.
        goal (str): Goal of the project manager.
        constraints (str): Constraints or limitations for the project manager.
    """

    name: str = "Eve"
    profile: str = "Project Manager"
    goal: str = (
        "break down tasks according to PRD/technical design, generate a task list, and analyze task "
        "dependencies to start with the prerequisite modules"
    )


    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.set_actions([])
