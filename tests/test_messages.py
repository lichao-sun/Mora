import unittest
from mora.messages import Message

class TestMessage(unittest.TestCase):
    def test_message(self):
        message = Message(id="1", sender="2", content="Hello")
        self.assertEqual(message.id, "1")
        self.assertEqual(message.sender, "2")
        self.assertEqual(message.content, "Hello")












if __name__ == '__main__':
    unittest.main()
