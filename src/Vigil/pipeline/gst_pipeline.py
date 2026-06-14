"""
GStreamer 视频管道模块

负责：
    1. 从 RTSP/IPC 摄像头拉流
    2. 解码视频帧
    3. 送入推理引擎
    4. 叠加检测结果到画面
    5. 推流或存储

管道架构:
    rtsp://camera → decode → tee
                              ├── appsink (取帧推理)
                              └── overlay → encode → rtsp(sink) / file
"""
import threading
from collections.abc import Callable

import numpy as np
from loguru import logger

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst
    Gst.init(None)
    GST_AVAILABLE = True
except (ImportError, ValueError) as e:
    Gst = None
    GLib = None
    GST_AVAILABLE = False
    logger.warning(f"GStreamer not available: {e}. Will use mock pipeline if needed.")


if GST_AVAILABLE:

    class VideoPipeline:
        """
        GStreamer 视频管道封装
        """

        def __init__(
            self,
            source_uri: str,
            inference_callback: Callable | None = None,
            fps: int = 15,
            buffer_size: int = 30,
        ):
            self.source_uri = source_uri
            self.inference_callback = inference_callback
            self.fps = fps
            self.buffer_size = buffer_size

            self.pipeline = None  # Gst.Pipeline
            self.appsink = None   # Gst.Element
            self.running = False
            self.loop = GLib.MainLoop()
            self.thread: threading.Thread | None = None

        def build_pipeline(self) -> str:
            """构建 GStreamer 管道字符串"""
            pipeline_str = f"""
                rtspsrc location={self.source_uri} latency=0 !
                queue ! rtph264depay ! h264parse ! avdec_h264 !
                videoconvert ! video/x-raw,format=RGB !
                tee name=t
                t. ! queue ! appsink name=inference_sink emit-signals=true sync=false
                t. ! queue ! videoconvert ! video/x-raw,format=I420 !
                      fpsdisplaysink video-sink=fakesink sync=false
            """
            return pipeline_str

        def build_from_string(self, pipeline_str: str = None):
            """从字符串构建管道"""
            if pipeline_str is None:
                pipeline_str = self.build_pipeline()
            self.pipeline = Gst.parse_launch(pipeline_str)
            self.appsink = self.pipeline.get_by_name("inference_sink")
            if self.appsink:
                self.appsink.connect("new-sample", self._on_new_sample)

            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)

        def _on_new_sample(self, appsink):
            """推理回调 —— 每帧触发"""
            sample = appsink.emit("pull-sample")
            if sample and self.inference_callback:
                buf = sample.get_buffer()
                caps = sample.get_caps()
                height = caps.get_structure(0).get_value("height")
                width = caps.get_structure(0).get_value("width")

                success, map_info = buf.map(Gst.MapFlags.READ)
                if success:
                    frame = np.ndarray(
                        shape=(height, width, 3),
                        dtype=np.uint8,
                        buffer=map_info.data,
                    )
                    self.inference_callback(frame.copy())
                    buf.unmap(map_info)

            return Gst.FlowReturn.OK

        def _on_bus_message(self, bus, message):
            t = message.type
            if t == Gst.MessageType.EOS:
                logger.info("Pipeline EOS")
                self.loop.quit()
            elif t == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                logger.error(f"GStreamer error: {err}, {debug}")
                self.loop.quit()

        def start(self):
            """启动管道（非阻塞）"""
            if self.pipeline is None:
                self.build_from_string()
            self.pipeline.set_state(Gst.State.PLAYING)
            self.running = True
            self.thread = threading.Thread(target=self.loop.run, daemon=True)
            self.thread.start()
            logger.info("Pipeline started")

        def stop(self):
            """停止管道"""
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
                self.loop.quit()
                self.running = False
                logger.info("Pipeline stopped")

else:
    # GStreamer 不可用时的占位
    VideoPipeline = None


class MockPipeline:
    """
    模拟管道 —— 用于无摄像头时的开发测试
    从本地视频文件或测试图片读取
    """

    def __init__(self, video_path: str = None):
        self.video_path = video_path
        self.callbacks: list = []

    def add_callback(self, callback: Callable):
        self.callbacks.append(callback)

    def run(self):
        """从本地视频文件逐帧读取"""
        import cv2
        cap = cv2.VideoCapture(self.video_path or 0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            for cb in self.callbacks:
                cb(frame)
        cap.release()

