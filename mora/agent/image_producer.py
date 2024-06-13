from mora.agent import Role
from mora.actions import GenerateImageWithTextAPI, GeneratePrompt
from mora.messages import Message
import asyncio

class ImageProducer(Role):
    name: str = "Mike"
    profile: str = "Image Producer"
    goal : str = "Improve the input text, and generate image with the generated prompt"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([GeneratePrompt, GenerateImageWithTextAPI])
        self._set_react_mode(react_mode="plan_and_act")

    async def _act(self):

        todo = self.todo

        msg = self.get_memories(k=1)[0]  # find the most k recent messages
        result = await todo.run(msg)
        print(result)
        msg=result
        msg.role=self.profile
        msg.cause_by=type(todo)

        self.memory.add(msg)


        return msg




