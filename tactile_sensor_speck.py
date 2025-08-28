import samna
import os
import sys
import time
from threading import Thread

class SpeckTac:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.graph.stop()
        print("SpeckTac shut down.")

    def __init__(self, save_dir="~/Speck_DVS_data", duration=2, width_proportion=0.75, height_proportion=0.75, streamer_endpoint="tcp://0.0.0.0:40001"):
        self.duration = duration
        self.width_proportion = width_proportion
        self.height_proportion = height_proportion
        self.streamer_endpoint = streamer_endpoint
        self.save_dir = os.path.expanduser(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.graph = samna.graph.EventFilterGraph()
        self.gui_process = None
        self.dk = None
        self.stopwatch = None

        self.open_device()
        self.start_visualizer()
        self.build_event_route()
        self.graph.start()
        self.enable_monitoring()

    def open_device(self):
        devices = [
            device for device in samna.device.get_unopened_devices()
            if device.device_type_name.startswith("Speck2f")
        ]
        assert devices, "No Speck2f device found."
        self.dk = samna.device.open_device(devices[0])
        self.stopwatch = self.dk.get_stop_watch()

    def start_visualizer(self):
        cmd = f"import samna, samnagui; samnagui.run_visualizer('{self.streamer_endpoint}', {self.width_proportion}, {self.height_proportion})"
        os_cmd = f"{sys.executable} -c \"{cmd}\""
        self.gui_process = Thread(target=os.system, args=(os_cmd,))
        self.gui_process.start()

    def build_event_route(self):
        _, _, streamer = self.graph.sequential(
            [self.dk.get_model_source_node(), "Speck2fDvsToVizConverter", "VizEventStreamer"]
        )
        config_source, _ = self.graph.sequential([samna.BasicSourceNode_ui_event(), streamer])
        streamer.set_streamer_endpoint(self.streamer_endpoint)
        if streamer.wait_for_receiver_count() == 0:
            raise Exception(f"Could not connect to visualizer on {self.streamer_endpoint}")
        config = samna.ui.VisualizerConfiguration(
            plots=[samna.ui.ActivityPlotConfiguration(128, 128, "DVS Layer")]
        )
        config_source.write([config])

    def enable_monitoring(self):
        config = samna.speck2f.configuration.SpeckConfiguration()
        config.dvs_layer.monitor_enable = True
        self.dk.get_model().apply_configuration(config)

    def record_once(self, filename="sample_000.bin"):
        sink = samna.graph.sink_from(self.dk.get_model_source_node())
        self.stopwatch.start(True)
        time.sleep(0.1)
        sink.clear_events()
        time.sleep(self.duration)
        events = sink.get_events()

        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, "wb") as f:
            for event in events:
                if isinstance(event, samna.speck2f.event.Spike):
                    f.write((event.x).to_bytes(4, byteorder='little'))
                    f.write((event.y).to_bytes(4, byteorder='little'))
                    f.write((event.feature).to_bytes(4, byteorder='little'))
                    f.write((event.timestamp).to_bytes(4, byteorder='little'))
        # print(f"Saved to {filepath}")

    def threaded_record_once(self, filename="sample_000.bin"):
        thread = Thread(target=self.record_once, args=(filename,))
        thread.start()
        return thread  # 可选返回，供调用方决定是否 join
