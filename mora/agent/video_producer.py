from mora.agent import Role
from mora.actions import GenerateImageWithText, GeneratePrompt, GenerateVideoWithImage
from mora.messages import Message
import asyncio
class VideoProducer(Role):
    name: str = "Mike"
    profile: str = "Image Producer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.actions = [GenerateImageWithText, GenerateVideoWithImage]
        self.set_actions(self.actions)
        self._set_react_mode(react_mode="by_order")

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





