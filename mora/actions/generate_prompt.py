from typing import Optional

from tenacity import retry, stop_after_attempt, wait_random_exponential

from mora.actions import Action
from mora.llm.azure_openai_api import AzureOpenAILLM
from mora.configs.llm_config import LLMConfig
from mora.messages import Message
from mora.llm.llm import BaseLLM

from mora.llm.ollama_api import OllamaLLM
from mora.llm.azure_openai_api import AzureOpenAILLM
llm_config = LLMConfig(
    api_type="ollama",
    base_url= "http://127.0.0.1:11434/api",
    model="llama3",

)

# llm_config = LLMConfig(
#     api_key=,
#     api_version=,
#     azure_endpoint=,
#     model=,
#
# )




PROMPT_TEMPLATE = """
NOTICE
Role:

# Context
## Input I want you to act as a prompt generator for artificial intelligence program. Your job is to provide detailed and creative descriptions from the {input} that will inspire unique and interesting images from the AI. Keep in mind that the AI is capable of understanding a wide range of language and can interpret abstract concepts, so feel free to be as imaginative and descriptive as possible. The prompt should not be too long.
{input}








"""


class GeneratePrompt(Action):

    name: str = "Generate Prompt"
    llm:Optional[BaseLLM]=None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.llm = AzureOpenAILLM(config=llm_config)
        self.llm = OllamaLLM(config=llm_config)


    async def _aask(self, prompt: str, system_msgs: Optional[list[str]] = None) -> str:
        """Append default prefix"""
        return await self.llm.aask(prompt, system_msgs,stream=False)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    async def generate_prompt(self, prompt) -> str:
        description = await self._aask(prompt)

        return description

    async def run(self, msg: Message) -> str:
        prompt = PROMPT_TEMPLATE.format(
            input=msg.content,

        )
        description_prompt = await self.generate_prompt(prompt)
        #description_prompt="Imagine a scene where the boundaries of time blur, set within the bustling confines of an ageless train station. At the heart of this temporal crossroads stands a man, an enigmatic figure who exudes an air of bygone elegance. He is the Timeless Traveler, draped in a tailored suit that whispers tales of a hundred years—a fabric interwoven with the essence of both the roaring twenties and the poised modernity of the future.  His suit, a tapestry of midnight blue, is accented with subtle pinstripes that seem to dance and shift with the station's ambient light. The suit clings to his form with a precision that suggests it was crafted by the hands of a master tailor from another era, one who understood the art of balancing classic style with an air of mystery.  Atop his head sits a hat, not just any hat, but a fedora that carries the weight of history in its brim. It's a deep charcoal gray, adorned with a band of silk the color of storm clouds just before the rain. The hat casts a shadow that partially obscures his eyes, adding to his enigmatic presence.  In his hand, he holds a cane, but this is no mere walking stick. It is an artifact that transcends its utilitarian purpose, becoming a symbol of the traveler's journey through time. The cane is crafted from polished mahogany, its handle carved into the shape of an intricate knot that defies the simplicity of its function. It is as if the cane itself is a key to unlocking the doors between decades and centuries.  The train station around him is a marvel of architectural fusion, where the steam-powered romanticism of the Victorian era meets the sleek lines of a future metropolis. Vaulted ceilings adorned with intricate frescoes loom overhead, while holographic timetables flicker with the promise of destinations both familiar and unfathomable.  Passengers from all walks of life and time itself weave around the Timeless Traveler, each absorbed in their own narratives. Some are adorned in garments that hark back to historical epochs, while others are clad in attire that seems to be spun from light and innovation.  In the background, trains resembling mechanical serpents glide silently along their tracks. Some appear as classic locomotives, puffing clouds of steam and echoing with the chug of industry, while others are sleek capsules of gleaming metal and glass, humming with the electricity of progress.  The air is filled with a symphony of sounds: the nostalgic whistle of departing trains, the murmur of a crowd that spans centuries, and the subtle, almost imperceptible hum of time itself bending around the presence of the Timeless Traveler.  As the AI processes this tableau, it is invited to capture not just the visual splendor of the scene but the palpable sense of wonder and the eternal dance between the past, present, and future. The image it conjures should be one that invites the viewer to step into a world where time is not a line, but a vast, beautiful expanse to be explored in all directions."
        new_msg=msg.model_copy()
        new_msg.content=description_prompt
        new_msg.sent_from=self.name
        return  new_msg
