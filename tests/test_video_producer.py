import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mora.agent.video_producer import VideoProducer
import numpy as np
import asyncio
def main(msg="This image captures a slender white rocket in the midst of launch, with a fiery exhaust trail, against a clear sky over a desolate desert landscape."):
    # role = SimpleCoder()
    role = VideoProducer()

    result = asyncio.run(role.run(msg))
    video_frames = [np.array(frame) for frame in result.image_content]
    video_frames.write_videofile("h.mp4", codec='libx264', fps=7)
    




if __name__ == "__main__":
    main()