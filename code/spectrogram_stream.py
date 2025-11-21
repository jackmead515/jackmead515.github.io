import cv2
import numpy as np
import torchaudio as ta
import torch as t
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

hls_source = "https://audio-orcasound-net.s3.amazonaws.com/rpi_orcasound_lab/hls/1763452821/live.m3u8"

source_pipeline = f'''
souphttpsrc location="{hls_source}" is-live=true \
    ! hlsdemux \
    ! queue \
    ! decodebin \
    ! audioconvert \
    ! audiobuffersplit output-buffer-duration=1/60 \
    ! appsink name=read emit-signals=true
'''.strip()

spec_transform = ta.transforms.Spectrogram(
    n_fft=1024,
    win_length=1024,
    hop_length=512,
    normalized=True,
    power=2.0
)
power_to_db = ta.transforms.AmplitudeToDB("power", 150.0)

spectrogram_duration = 5.0  # seconds
sample_rate = 48000  # Hz
n_fft = 1024
hop_length = 512

spectrogram_image_height = n_fft // 2 + 1
spectrogram_image_width = int((spectrogram_duration * sample_rate) / hop_length)

spectrogram_image = np.zeros((spectrogram_image_height, spectrogram_image_width), dtype=np.uint8)

print(f"Spectrogram image size: {spectrogram_image_width}x{spectrogram_image_height}")

video_frame = np.zeros((480, 640), dtype=np.uint8)
scale_w = 640 / spectrogram_image_width
scale_h = 480 / spectrogram_image_height
scale = min(scale_w, scale_h)
video_width = int(spectrogram_image_width * scale)
video_height = int(spectrogram_image_height * scale)

print(f"Resized spectrogram size: {video_width}x{video_height}")

sink_pipeline = f'''
appsrc name=video emit-signals=true format=time is-live=true \
appsrc name=audio emit-signals=true format=time is-live=true \
matroskamux name=mux ! filesink location="output.mkv" \
video. ! queue ! videoconvert ! video/x-raw,format=I420 ! x264enc tune=zerolatency speed-preset=ultrafast ! h264parse config-interval=-1 ! mux. \
audio. ! queue ! audioconvert ! audio/x-raw,format=S16LE ! flacenc ! mux.
'''.strip()

loop = GLib.MainLoop.new(None, False)

source_pipeline = Gst.parse_launch(source_pipeline)
sink_pipeline = Gst.parse_launch(sink_pipeline)

audio_source = source_pipeline.get_by_name("read")
video_sink = sink_pipeline.get_by_name("video")
audio_sink = sink_pipeline.get_by_name("audio")

video_caps = Gst.Caps.from_string(f"video/x-raw,format=GRAY8,width=640,height=480,framerate=60/1")
audio_caps = Gst.Caps.from_string("audio/x-raw,format=F32LE,layout=interleaved,rate=48000,channels=2")

video_sink.set_property('caps', video_caps)
audio_sink.set_property('caps', audio_caps)

pts = 0

def on_audio_sample(sink):
    global spec_transform, power_to_db, spectrogram_image, video_frame, pts

    sample = sink.emit("pull-sample")
    buffer = sample.get_buffer()
    caps = sample.get_caps()
    success, map_info = buffer.map(Gst.MapFlags.READ)

    if not success:
        return Gst.FlowReturn.ERROR

    tensor = np.frombuffer(map_info.data, dtype=np.float32).copy()
    tensor = tensor.reshape(-1, 2).T
    tensor = t.from_numpy(tensor).float()
    tensor = power_to_db(spec_transform(tensor))
    tensor += 100.0
    tensor = (tensor / 150.0) * 255.0
    tensor = tensor.numpy().astype(np.uint8)
    tensor = tensor[0, :]
    tensor = np.flipud(tensor)

    spectrogram_image[:, :-tensor.shape[1]] = spectrogram_image[:, tensor.shape[1]:]
    spectrogram_image[:, -tensor.shape[1]:] = tensor

    resized_spec = cv2.resize(spectrogram_image, (video_width, video_height))
    
    # plcae resized_spec in the center of video_frame
    y_offset = (480 - video_height) // 2
    x_offset = (640 - video_width) // 2
    video_frame[y_offset:y_offset+video_height, x_offset:x_offset+video_width] = resized_spec
    
    frame_buffer = Gst.Buffer.new_wrapped(video_frame.tobytes())
    frame_buffer.pts = pts
    frame_buffer.duration = buffer.duration

    audio_buffer = Gst.Buffer.new_wrapped(map_info.data)
    audio_buffer.pts = pts
    audio_buffer.duration = buffer.duration

    pts += frame_buffer.duration

    video_sink.emit("push-buffer", frame_buffer)
    audio_sink.emit("push-buffer", audio_buffer)

    audio_buffer.unmap(map_info)
    video_frame[:] = 0

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

audio_source.connect("new-sample", lambda sink: on_audio_sample(sink))

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