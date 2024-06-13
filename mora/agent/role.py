from enum import Enum
from typing import TYPE_CHECKING, Iterable, Optional, Type, Union, Any

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator

from mora.actions import Action
from mora.messages import Message
from mora.messages import MessageQueue
from mora.memory import Memory
from mora.actions import UserRequirement
from mora.actions import GenerateSoPs
from mora.human_input.base_human_input import HumanProvider
from mora.configs.llm_config import LLMConfig
from mora.llm.ollama_api import OllamaLLM
from mora.llm.azure_openai_api import AzureOpenAILLM
# llm_config = LLMConfig(
#     api_type="ollama",
#     base_url= "http://127.0.0.1:11434/api",
#     model="llama3",
#
# )

llm_config = LLMConfig(
    api_key='',
    api_version="",
    azure_endpoint="",
    model='',

)
PREFIX_TEMPLATE = """You are a {profile}, named {name}, your goal is {goal}. """
CONSTRAINT_TEMPLATE = "the constraint is {constraints}. "
STATE_TEMPLATE_HUMAN_SOPs = """You can decide which stage you should enter or stay in based on human requirements,and instruction.
Please note that only the overall SOPs is {SOPs}
Please note that only the text between the first and second "===" is a human instruction,which indicate the next step should go or back to the previous stage. The text between the first and second "|" is the previous stage.
You should choose the next stage based on human instruction and previous stages.
===
{human_input}
===

Your previous stage: |{previous_state}|

Now choose one of the following stages you need to go to in the next step:
{states}

Just answer a number between 0-{n_states}, and choose the most suitable stage according to the human input.
Please note that the answer only needs a number, no need to add any other text.
If the final step is done, and the instruction is continue, return -1.
Do not answer anything else, and do not add any other information in your answer.
"""
STATE_TEMPLATE_HUMAN = """You can decide which stage you should enter or stay in based on human requirements.
Please note that only the text between the first and second "===" is a human instruction,which indicate the next step should go or back to the previous stage. The text between the first and second "|" is the previous stage.
You should choose the next stage based on human instruction and previous stages.
===
{human_input}
===

Your previous stage: |{previous_state}|

Now choose one of the following stages you need to go to in the next step:
{states}

Just answer a number between 0-{n_states}, and choose the most suitable stage according to the human input.
Please note that the answer only needs a number, no need to add any other text.
If the final step is done, and the instruction is continue, return -1.
Do not answer anything else, and do not add any other information in your answer.
"""
STATE_TEMPLATE = """Here are your conversation records. You can decide which stage you should enter or stay in based on these records.
Please note that only the text between the first and second "===" is information about completing tasks and should be used as reference. Decide which stage you should enter or stay in based on the given previous stage.
===
{history}
===

Your previous stage: {previous_state}

Now choose one of the following stages you need to go to in the next step:
{states}

Just answer a number between 0-{n_states}, choose the most suitable stage according to the understanding of the conversation. Do not repeat the previous stage.
Please note that the answer only needs a number, no need to add any other text.
If you think you have completed your goal and don't need to go to any of the stages, return -1.
Do not answer anything else, and do not add any other information in your answer.
"""

ROLE_TEMPLATE = """Your response should be based on the previous conversation history and the current conversation stage.

## Current conversation stage
{state}

## Conversation history
{history}
{name}: {result}
"""


def get_class_name(cls) -> str:
    """Return class name"""
    return f"{cls.__module__}.{cls.__name__}"


def any_to_str(val: Any) -> str:
    """Return the class name or the class name of the object, or 'val' if it's a string type."""
    if isinstance(val, str):
        return val
    elif not callable(val):
        return get_class_name(type(val))
    else:
        return get_class_name(val)


class RoleReactMode(str, Enum):
    REACT = "react"
    BY_ORDER = "by_order"
    PLAN_AND_ACT = "plan_and_act"

    @classmethod
    def values(cls):
        return [item.value for item in cls]


class Role(BaseModel):
    """Role/Agent"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = ""
    profile: str = ""
    goal: str = ""
    constraints: str = ""
    desc: str = ""
    is_human: bool = False
    SOPs: list[str] = []
    state: int = Field(default=-1)
    role_id: str = ""
    states: list[str] = []

    # scenarios to set action system_prompt:
    #   1. `__init__` while using Role(actions=[...])
    #   2. add action to role while using `role.set_action(action)`
    #   3. set_todo while using `role.set_todo(action)`
    #   4. when role.system_prompt is being updated (e.g. by `role.system_prompt = "..."`)
    # Additional, if llm is not set, we will use role's llm
    actions: list[SerializeAsAny[Action]] = Field(default=[], validate_default=True)
    state: int = Field(default=-1)  # -1 indicates initial or termination state where todo is None
    todo: Action = Field(default=None, exclude=True)
    news: list[Type[Message]] = Field(default=[], exclude=True)
    msg_buffer: MessageQueue = Field(
        default_factory=MessageQueue, exclude=True
    )
    memory: Memory = Field(default_factory=Memory)
    watch: set[str] = Field(default_factory=set)
    # builtin variables
    recovered: bool = False  # to tag if a recovered role
    latest_observed_msg: Optional[Message] = None  # record the latest observed message when interrupted

    __hash__ = object.__hash__  # support Role as hashable type in `Environment.members`

    @model_validator(mode="after")
    def validate_role_extra(self):
        self._process_role_extra()
        return self

    def _process_role_extra(self):
        kwargs = self.model_extra or {}
        if self.is_human:
            self.human = HumanProvider()
        self.max_react_loop=10
        self._check_actions()
        self.llm = AzureOpenAILLM(config=llm_config)
        self._watch(kwargs.pop("watch", [UserRequirement]))




        if self.latest_observed_msg:
            self.recovered = True

    @property
    def todo(self) -> Action:
        """Get action to do"""
        return self.todo

    def set_todo(self, value: Optional[Action]):

        self.todo = value

    @property
    def prompt_schema(self):
        """Prompt schema: json/markdown"""
        return self.config.prompt_schema

    @property
    def project_name(self):
        return self.config.project_name

    @project_name.setter
    def project_name(self, value):
        self.config.project_name = value

    @property
    def project_path(self):
        return self.config.project_path

    def _reset(self):
        self.states = []
        self.actions = []

    @property
    def _setting(self):
        return f"{self.name}({self.profile})"

    def _check_actions(self):
        """Check actions and set llm and prefix for each action."""
        self.set_actions(self.actions)
        return self

    @property
    def history(self) -> list[Message]:
        return self.memory.get()

    def put_message(self, message):
        """Place the message into the Role object's private message buffer."""
        if not message:
            return
        self.msg_buffer.push(message)

    def get_memories(self, k=0) -> list[Message]:
        """A wrapper to return the most recent k memories of this role, return all when k=0"""
        return self.memory.get(k=k)
    def set_action(self, action: Action):
        """Add action to the role."""
        self.set_actions([action])

    def set_actions(self, actions: list[Union[Action, Type[Action]]]):
        """Add actions to the role.

        Args:
            actions: list of Action classes or instances
        """
        self._reset()
        for action in actions:
            i = action()
            self.actions.append(i)
            self.states.append(f"{len(self.actions) - 1}. {action}")

    def _set_react_mode(self, react_mode: str, max_react_loop: int = 1, auto_run: bool = True):
        """Set strategy of the Role reacting to observed Message. Variation lies in how
        this Role elects action to perform during the _think stage, especially if it is capable of multiple Actions.

        Args:
            react_mode (str): Mode for choosing action during the _think stage, can be one of:
                        "react": standard think-act loop in the ReAct paper, alternating thinking and acting to solve the task, i.e. _think -> _act -> _think -> _act -> ...
                                 Use llm to select actions in _think dynamically;
                        "by_order": switch action each time by order defined in _init_actions, i.e. _act (Action1) -> _act (Action2) -> ...;
                        "plan_and_act": first plan, then execute an action sequence, i.e. _think (of a plan) -> _act -> _act -> ...
                                        Use llm to come up with the plan dynamically.
                        Defaults to "react".
            max_react_loop (int): Maximum react cycles to execute, used to prevent the agent from reacting forever.
                                  Take effect only when react_mode is react, in which we use llm to choose actions, including termination.
                                  Defaults to 1, i.e. _think -> _act (-> return result and end)
        """

        self.react_mode = react_mode

    def _watch(self, actions: Iterable[Type[Action]] | Iterable[Action]):
        """Watch Actions of interest. Role will select Messages caused by these Actions from its personal message
        buffer during _observe.
        """
        self.watch = {any_to_str(t) for t in actions}

    def is_watch(self, caused_by: str):
        return caused_by in self.watch

    def _set_state(self, state: int):
        """Update the current state."""
        self.state = state

        self.set_todo(self.actions[self.state] if state >= 0 else None)

    def _get_prefix(self):
        """Get the role prefix"""
        if self.desc:
            return self.desc

        prefix = PREFIX_TEMPLATE.format(**{"profile": self.profile, "name": self.name, "goal": self.goal})

        if self.constraints:
            prefix += CONSTRAINT_TEMPLATE.format(**{"constraints": self.constraints})


        return prefix
    async def _get_SOPs(self):
        actions= []
        for action in self.actions:
            actions.append(action.name)
        msg = Message(content=str(actions))

        action = GenerateSoPs()
        SOPs_message = await action.run(msg, goal=self.goal, n_actions=len(actions)-1)
        self.SOPs = eval(SOPs_message.content)

    async def _think(self) -> bool:
        """Consider what to do and decide on the next course of action. Return false if nothing can be done."""
        if len(self.actions) == 1:
            # If there is only one action, then only this one can be performed
            self._set_state(0)

            return True
        if self.is_human and self.state >= 0:
            human_input = await self.human.aask("Please input your choice:")
            if human_input == "exit":
                self._set_state(-1)
                return False
        if self.recovered and self.state >= 0:
            self._set_state(self.state)  # action to run from recovered state
            self.recovered = False  # avoid max_react_loop out of work
            return True

        prompt = self._get_prefix()


        if self.is_human and self.state >= 0:
            prompt += STATE_TEMPLATE_HUMAN.format(
                history=self.history,
                states="\n".join(self.states),
                n_states=len(self.states) - 1,
                previous_state=self.state,
                human_input=human_input,
            )
        else:
            prompt += STATE_TEMPLATE.format(
                history=self.history,
                states="\n".join(self.states),
                n_states=len(self.states) - 1,
                previous_state=self.state,
            )

        next_state = await self.llm.aask(prompt)
        if self.state==int(next_state) and next_state!=len(self.states)-1:
            next_state=self.state+1

        if (not next_state.isdigit() and next_state != "-1") or int(next_state) not in range(-1, len(self.states)):

            next_state = -1
        else:
            next_state = int(next_state)

        self._set_state(next_state)
        return True

    async def _act(self) -> Message:

        response = await self.todo.run(self.history)

        if isinstance(response, Message):
            msg = response
        else:
            msg = Message(content=response, role=self.profile, cause_by=self.rc.todo, sent_from=self)
        self.memory.add(msg)

        return msg

    async def _observe(self, ignore_memory=False) -> int:
        """Prepare new messages for processing from the message buffer and other sources."""
        # Read unprocessed messages from the msg buffer.
        news = []
        if self.recovered:
            news = [self.latest_observed_msg] if self.latest_observed_msg else []
        if not news:
            news = self.msg_buffer.pop_all()
        # Store the read messages in your own memory to prevent duplicate processing.
        old_messages = [] if ignore_memory else self.memory.get()
        self.memory.add_batch(news)
        # Filter out messages of interest.
        for n in news:
            print(n.cause_by, self.watch, self.name, n.send_to)
        self.news = [
            n for n in news if (n.cause_by in self.watch or self.name in n.send_to) and n not in old_messages
        ]
        self.latest_observed_msg = self.news[-1] if self.news else None  # record the latest observed msg

        # Design Rules:
        # If you need to further categorize Message objects, you can do so using the Message.set_meta function.
        # msg_buffer is a receiving buffer, avoid adding message data and operations to msg_buffer.
        news_text = [f"{i.role}: {i.content[:20]}..." for i in self.news]

        return len(self.news)

    async def _react(self) -> Message:
        """Think first, then act, until the Role _think it is time to stop and requires no more todo.
        This is the standard think-act loop in the ReAct paper, which alternates thinking and acting in task solving, i.e. _think -> _act -> _think -> _act -> ...
        Use llm to select actions in _think dynamically
        """
        actions_taken = 0
        rsp = Message(content="No actions taken yet", cause_by=Action)  # will be overwritten after Role _act
        while actions_taken < self.max_react_loop:
            # think
            await self._think()
            if self.todo is None:
                break
            # act

            rsp = await self._act()
            actions_taken += 1
        return rsp  # return output from the last action

    async def _act_by_order(self) -> Message:
        """switch action each time by order defined in _init_actions, i.e. _act (Action1) -> _act (Action2) -> ..."""
        start_idx = self.state if self.state >= 0 else 0  # action to run from recovered state
        rsp = Message(content="No actions taken yet")  # return default message if actions=[]
        for i in range(start_idx, len(self.states)):
            self._set_state(i)
            rsp = await self._act()
        return rsp  # return output from the last action

    async def _plan_and_act(self) -> Message:
        if self.is_human:
            state = self.state
            while state in self.SOPs:
                human_input = await self.human.aask("Please input your choice:")
                if human_input == "exit":
                    break

                prompt = self._get_prefix()

                prompt += STATE_TEMPLATE_HUMAN_SOPs.format(
                        SOPs=self.SOPs,
                        history=self.history,
                        states="\n".join(self.states),
                        n_states=len(self.states) - 1,
                        previous_state=self.state,
                        human_input=human_input,
                    )


                next_state = await self.llm.aask(prompt)
                if (not next_state.isdigit() and next_state != "-1") or int(next_state) not in range(-1,
                                                                                                     len(self.states)):

                    next_state = -1
                else:
                    next_state = int(next_state)
                self._set_state(next_state)
                state = self.state
                rsp = await self._act()
            return rsp

        else:
            for i in self.SOPs:
                if i==-1:
                    break
                self._set_state(i)
                rsp = await self._act()
            return rsp

    async def _act_on_task(self, current_task):
        """Taking specific action to handle one task in plan

        Args:
            current_task (Task): current task to take on

        Raises:
            NotImplementedError: Specific Role must implement this method if expected to use planner

        Returns:
            TaskResult: Result from the actions
        """
        raise NotImplementedError

    async def react(self) -> Message:
        """Entry to one of three strategies by which Role reacts to the observed Message"""
        if self.react_mode == RoleReactMode.REACT:
            rsp = await self._react()
        elif self.react_mode == RoleReactMode.BY_ORDER:
            rsp = await self._act_by_order()
        elif self.react_mode == RoleReactMode.PLAN_AND_ACT:
            rsp = await self._plan_and_act()
        else:
            raise ValueError(f"Unsupported react mode: {self.react_mode}")
        self._set_state(state=-1)  # current reaction is complete, reset state to -1 and todo back to None
        return rsp

    async def run(self, with_message=None) -> Message | None:
        """Observe, and think and act based on the results of the observation"""
        await self._get_SOPs()
        if with_message:
            msg = None
            if isinstance(with_message, str):
                msg = Message(content=with_message)
            elif isinstance(with_message, Message):
                msg = with_message
            elif isinstance(with_message, list):
                msg = Message(content="\n".join(with_message))
            if not msg.cause_by:
                msg.cause_by = UserRequirement
            self.put_message(msg)
        if not await self._observe():
            # If there is no new information, suspend and wait

            return
        rsp = await self.react()

        # Reset the next action to be taken.
        self.set_todo(None)

        return rsp

    async def think(self) -> Action:
        """
        Export SDK API, used by AgentStore RPC.
        The exported `think` function
        """
        await self._observe()  # For compatibility with the old version of the Agent.
        await self._think()
        return self.todo

    async def act(self):
        """
        Export SDK API, used by AgentStore RPC.
        The exported `act` function
        """
        msg = await self._act()
        return Message(content=msg.content)
