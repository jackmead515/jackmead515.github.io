import subprocess
import time

import cv2
import numpy as np
import torch as t
import torchaudio as ta
from ultralytics import YOLO
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

model = YOLO("cfd-yolov12x-1.00.pt", verbose=False, task="detect")

fps = 15
spectrogram_duration = 5.0  # seconds
sample_rate = 44100  # Hz
n_fft = 1024
hop_length = 512
spect_image_height = n_fft // 2 + 1
spect_image_width = int((spectrogram_duration * sample_rate) / hop_length)
spect_image = np.zeros((spect_image_height, spect_image_width), dtype=np.float32)

# there is some introduced distortion. But for some reason I 
# have to use some standard dimensions for the spectrogram
# otherwise gstreamer freaks out.
spect_video_width = 480
spect_video_height = 320

video_width = 1920
video_height = 1080

power_to_db = ta.transforms.AmplitudeToDB("power", 150.0)

spec_transform = ta.transforms.Spectrogram(
    n_fft=n_fft,
    win_length=n_fft,
    hop_length=hop_length,
    normalized=True,
    power=2.0
)

# grab the HLS source URL using yt-dlp
hls_source = subprocess.run('yt-dlp --get-url "https://www.youtube.com/watch?v=UFA_SYoLqtk"', shell=True, capture_output=True, text=True).stdout.strip()

# we need to source from souphttpsrc since youtube uses https
# then HLS demux the stream
# then tsdemux to get audio and video streams separately
# We then pipe them into seperate threads using tee elements
# Each branch has a appsink to read the decoded raw data into python
source_pipeline = f'''
appsink name=videoread emit-signals=true \
appsink name=audioread emit-signals=true \
souphttpsrc location="{hls_source}" is-live=true \
! hlsdemux \
! tsdemux name=demux \
demux. \
    ! tee name=video_tee \
    video_tee. \
    ! queue \
    ! h264parse \
    ! avdec_h264 \
    ! videoconvert \
    ! videorate \
    ! video/x-raw,format=RGB,framerate={fps}/1 \
    ! videoread. \
demux. \
    ! tee name=audio_tee \
    audio_tee. \
    ! queue \
    ! aacparse \
    ! faad \
    ! audioconvert \
    ! audioresample \
    ! audiobuffersplit output-buffer-duration=1/{fps} \
    ! audioread.
'''.strip()

# the compositor sinks the processed video and spectrogram
# onto a single video output
sink_pipeline = f'''
appsrc name=videowrite emit-signals=true format=time is-live=true \
appsrc name=audiowrite emit-signals=true format=time is-live=true \
appsrc name=spectwrite emit-signals=true format=time is-live=true \
matroskamux name=mux \
    ! filesink location="output.mkv" \
compositor name=mix background=black sink_1::xpos=20 sink_1::ypos={video_height - spect_video_height - 20} sink_1::alpha=0.9 \
    ! video/x-raw,width={video_width},height={video_height} \
    ! videoconvert \
    ! videorate \
    ! video/x-raw,framerate={fps}/1,format=I420 \
    ! x264enc tune=zerolatency speed-preset=ultrafast \
    ! h264parse config-interval=-1 \
    ! mux. \
videowrite. \
    ! queue \
    ! videoconvert \
    ! mix.sink_0 \
spectwrite. \
    ! queue \
    ! videoconvert \
    ! mix.sink_1 \
audiowrite. \
    ! queue \
    ! audioconvert \
    ! flacenc \
    ! mux.
'''

loop = GLib.MainLoop.new(None, False)

source_pipeline = Gst.parse_launch(source_pipeline)
sink_pipeline = Gst.parse_launch(sink_pipeline)

video_read = source_pipeline.get_by_name("videoread")
audio_read = source_pipeline.get_by_name("audioread")

video_write = sink_pipeline.get_by_name("videowrite")
audio_write = sink_pipeline.get_by_name("audiowrite")
spect_write = sink_pipeline.get_by_name("spectwrite")

video_caps = Gst.Caps.from_string(f"video/x-raw,format=RGB,width={video_width},height={video_height},framerate={fps}/1")
audio_caps = Gst.Caps.from_string("audio/x-raw,format=S16LE,layout=interleaved,rate=44100,channels=2")
spect_caps = Gst.Caps.from_string(f"video/x-raw,format=RGB,width={spect_video_width},height={spect_video_height},framerate={fps}/1")

print('video caps:', video_caps.to_string())
print('audio caps:', audio_caps.to_string())
print('spect caps:', spect_caps.to_string())

video_write.set_property('caps', video_caps)
audio_write.set_property('caps', audio_caps)
spect_write.set_property('caps', spect_caps)

def on_video_sample(sink):
    sample = sink.emit("pull-sample")
    buffer = sample.get_buffer()
    success, map_info = buffer.map(Gst.MapFlags.READ)

    if not success:
        return Gst.FlowReturn.ERROR

    frame = np.frombuffer(map_info.data, dtype=np.uint8)
    frame = frame.reshape((video_height, video_width, 3))
    
    results = model(frame, imgsz=640, verbose=False)
    image = results[0].plot(
        color_mode="class",
        probs=False
    )
    image = cv2.resize(image, (video_width, video_height))

    frame_buffer = Gst.Buffer.new_wrapped(image.tobytes())
    frame_buffer.pts = buffer.pts
    frame_buffer.duration = buffer.duration

    video_write.emit("push-buffer", frame_buffer)

    buffer.unmap(map_info)

    return Gst.FlowReturn.OK


def on_audio_sample(sink):
    global spect_image

    sample = sink.emit("pull-sample")
    buffer = sample.get_buffer()
    success, map_info = buffer.map(Gst.MapFlags.READ)

    if not success:
        return Gst.FlowReturn.ERROR

    data = map_info.data
    
    # the caps say S16LE, so we interpret the buffer as int16
    tensor = np.frombuffer(data, dtype=np.int16).copy()

    # this is two channel audio, reshape accordingly
    tensor = tensor.reshape(-1, 2).T

    # use only the first channel
    tensor = tensor[0, :]
    
    # generate spectrogram
    tensor = power_to_db(spec_transform(t.from_numpy(tensor).float()))

    # flip the spectrogram vertically so the high frequencies are at the top
    tensor = np.flipud(tensor.numpy())

    # crop part of the spectrogram into a buffer.
    # this is faster than recomputing the entire spectrogram each time
    spect_image[:, :-tensor.shape[1]] = spect_image[:, tensor.shape[1]:]
    spect_image[:, -tensor.shape[1]:] = tensor

    # normalize spect_image to 0-255
    norm_image = spect_image - spect_image.min()
    norm_image = norm_image / (norm_image.max() + 1e-6)
    norm_image = norm_image * 255.0
    norm_image = norm_image.astype(np.uint8)
    
    # apply color map. Inferno looks nice!
    norm_image = cv2.applyColorMap(norm_image, cv2.COLORMAP_INFERNO)
    norm_image = cv2.cvtColor(norm_image, cv2.COLOR_BGR2RGB)

    # cut off the upper half. Since the AAC codec applies
    # Spectoral Band Replication (SBR), the higher frequencies
    # are just copies of the lower frequencies.
    norm_image = norm_image[spect_image_height//2:, :]

    # resize to a standardize size. Some slight distortion is ok.
    norm_image = cv2.resize(norm_image, (spect_video_width, spect_video_height), interpolation=cv2.INTER_AREA)

    video_buffer = Gst.Buffer.new_wrapped(norm_image.tobytes())
    video_buffer.pts = buffer.pts
    video_buffer.duration = buffer.duration

    audio_buffer = Gst.Buffer.new_wrapped(data)
    audio_buffer.pts = buffer.pts
    audio_buffer.duration = buffer.duration

    audio_write.emit("push-buffer", audio_buffer)
    spect_write.emit("push-buffer", video_buffer)

    buffer.unmap(map_info)

    return Gst.FlowReturn.OK


def on_message(bus, message):
    if message.type == Gst.MessageType.EOS:
        print("End-Of-Stream reached.")
        loop.quit()
    elif message.type == Gst.MessageType.ERROR:
        print(f"GStreamer Source Error: {message.parse_error()}")
        loop.quit()
    elif message.type == Gst.MessageType.WARNING:
        print(f"GStreamer Source Warning: {message.parse_warning()}")


source_bus = source_pipeline.get_bus()
source_bus.add_signal_watch()
source_bus.connect("message", on_message)

sink_bus = sink_pipeline.get_bus()
sink_bus.add_signal_watch()
sink_bus.connect("message", on_message)

video_read.connect("new-sample", on_video_sample)
audio_read.connect("new-sample", on_audio_sample)

sink_pipeline.set_state(Gst.State.PLAYING)
source_pipeline.set_state(Gst.State.PLAYING)

try:
    loop.run()
except KeyboardInterrupt:
    print("Interrupted by user, stopping...")
except Exception:
    print("Failed during processing")
finally:
    source_pipeline.set_state(Gst.State.NULL)
    sink_pipeline.set_state(Gst.State.NULL)