from mora.actions import GenerateImageWithText
from mora.messages import Message
import pytest


@pytest.mark.asyncio
async def test_Generate_Image_With_Text():
    msg = Message(content="a dog")
    action= GenerateImageWithText()
    output_msg = await action.run(msg)
    assert output_msg.image_content is not None
    output_msg.image_content.show()

if __name__ == '__main__':
    test_Generate_Image_With_Text()