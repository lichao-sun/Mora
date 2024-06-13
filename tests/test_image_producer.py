import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))   

from mora.agent.image_producer import ImageProducer

from mora.actions import GenerateSoPs
from mora.messages import Message
import pytest
import asyncio
def main(msg="This image captures a slender white rocket in the midst of launch, with a fiery exhaust trail, against a clear sky over a desolate desert landscape."):
    # role = SimpleCoder()




    role = ImageProducer(is_human=True)

    result = asyncio.run(role.run(msg))
    result.image_content.save("generated.png")
    




if __name__ == "__main__":
    main()