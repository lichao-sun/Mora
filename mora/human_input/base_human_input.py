class HumanProvider():
    """Humans provide themselves as a 'model', which actually takes in human input as its response.
    This enables replacing LLM anywhere in the framework with a human, thus introducing human interaction
    """

    def ask(self, msg: str) -> str:
        print("Now it's your turn to provide human input.")
        rsp = input(msg)
        if rsp in ["exit", "quit"]:
            exit()
        return rsp

    async def aask(
        self,
        msg: str,

    ) -> str:
        return self.ask(msg)

