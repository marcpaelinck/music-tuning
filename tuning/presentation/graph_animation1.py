import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from tuning.common.constants import Folder, InstrumentGroupName
from tuning.common.utils import get_path

# Fixing random state for reproducibility
np.random.seed(19680801)

plt.rcParams["animation.ffmpeg_path"] = "C:\\Users\\marcb\\Desktop\\ffmpeg\\bin\\ffmpeg.exe"
plt.rcParams["backend"] = "Agg"

metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
writer = FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure()
(l,) = plt.plot([], [], "k-o")

plt.xlim(-5, 5)
plt.ylim(-5, 5)

x0, y0 = np.array([0.0]), np.array([0.0])

output = get_path(InstrumentGroupName.TEST, Folder.SETTINGS, "writer_test.mp4")

with writer.saving(fig, output, 100):
    for i in range(100):
        x0 += 0.1 * np.random.randn()
        y0 += 0.1 * np.random.randn()
        l.set_data(x0, y0)
        writer.grab_frame()
