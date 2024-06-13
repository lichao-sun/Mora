from pydantic import BaseModel, ConfigDict


class Action(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    name: str = ""
    prefix: str = ""

    @property
    def project_name(self):
        return self.config.project_name

    @project_name.setter
    def project_name(self, value):
        self.config.project_name = value

    @property
    def project_path(self):
        return self.config.project_path

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()

    async def run(self, *args, **kwargs):
        """Run action"""
        raise NotImplementedError("The run method should be implemented in a subclass.")
