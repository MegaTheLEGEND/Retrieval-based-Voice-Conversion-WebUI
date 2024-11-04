import os
import sys
from dotenv import load_dotenv

load_dotenv()

os.environ["OMP_NUM_THREADS"] = "4"
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

current_dir = os.getcwd()
sys.path.append(current_dir)
import multiprocessing

stream_latency = -1


def print_message(message, *args):
    if len(args) == 0:
        print(message)
    else:
        print(message % args)


def phase_vocoder(signal_a, signal_b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(signal_a * window)
    fb = torch.fft.rfft(signal_b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = signal_a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    phase_diff = phib - phia
    phase_diff = phase_diff - 2 * np.pi * torch.floor(phase_diff / (2 * np.pi) + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(signal_a) + phase_diff
    t = torch.arange(n).unsqueeze(-1).to(signal_a) / n
    result = signal_a * (fade_out ** 2) + signal_b * (fade_in ** 2) + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    return result

    
class Harvest(multiprocessing.Process):
    def __init__(self, input_queue, output_queue):
        multiprocessing.Process.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        import numpy as np
        import pyworld

        while True:
            idx, x, res_f0, n_cpu, ts = self.input_queue.get()
            f0, t = pyworld.harvest(
                x.astype(np.double),
                fs=16000,
                f0_ceil=1100,
                f0_floor=50,
                frame_period=10,
            )
            res_f0[idx] = f0
            if len(res_f0.keys()) >= n_cpu:
                self.output_queue.put(ts)


if __name__ == "__main__":
    import json
    import multiprocessing
    import re
    import threading
    import time
    import traceback
    from multiprocessing import Queue, cpu_count
    from queue import Empty

    import librosa
    from tools.torchgate import TorchGate
    import numpy as np
    import PySimpleGUI as sg
    import sounddevice as sd
    import torch
    import torch.nn.functional as F
    import torchaudio.transforms as tat

    import tools.rvc_for_realtime as rvc_for_realtime
    from i18n.i18n import I18nAuto
    from configs.config import Config

    i18n = I18nAuto()

    current_dir = os.getcwd()
    input_queue = Queue()
    output_queue = Queue()
    n_cpu = min(cpu_count(), 8)
    for _ in range(n_cpu):
        Harvest(input_queue, output_queue).start()

    class GUIConfig:
        def __init__(self) -> None:
            self.pth_path: str = ""
            self.index_path: str = ""
            self.pitch: int = 0
            self.samplerate: int = 40000
            self.block_time: float = 1.0  # seconds
            self.buffer_num: int = 1
            self.threshold: int = -60
            self.crossfade_time: float = 0.05
            self.extra_time: float = 2.5
            self.I_noise_reduce = False
            self.O_noise_reduce = False
            self.rms_mix_rate = 0.0
            self.index_rate = 0.3
            self.n_cpu = min(n_cpu, 6)
            self.f0method = "harvest"
            self.sg_input_device = ""
            self.sg_output_device = ""

    class GUI:
        def __init__(self):
            self.history_file = "configs/recent_pairs.json"  # Store history here
            self.recent_pairs = []  # This will hold the history of model-index pairs
            self.load_history()  # Load history on startup
            self.gui_config = GUIConfig()
            self.config = Config()
            self.flag_vc = False
            self.function = "vc"
            self.delay_time = 0
            self.launch()
                
                
                
        def load_history(self):
            """Load the recent model-index pairs from history file."""
            if os.path.exists(self.history_file):
                with open(self.history_file, "r") as file:
                    self.recent_pairs = json.load(file)
            else:
                self.recent_pairs = []

        def save_to_history(self, pth_path, index_path):
            """Save the current model and index pair to history."""
            pair = {"pth_path": pth_path, "index_path": index_path}
            if pair not in self.recent_pairs:
                self.recent_pairs.append(pair)
                if len(self.recent_pairs) > 10:  # Limit the history to 10 items
                    self.recent_pairs.pop(0)
                with open(self.history_file, "w") as file:
                    json.dump(self.recent_pairs, file)
        
        
        
        
        def load_config(self):
            """Load the configuration and recent model-index pairs."""
            # Get the list of available input and output devices
            input_devices, output_devices, _, _ = self.get_devices()

            # Load the main config file
            try:
                with open("configs/config.json", "r") as j:
                    data = json.load(j)
                    data["sr_model"] = data["sr_type"] == "sr_model"
                    data["sr_device"] = data["sr_type"] == "sr_device"
                    data["pm"] = data["f0method"] == "pm"
                    data["harvest"] = data["f0method"] == "harvest"
                    data["crepe"] = data["f0method"] == "crepe"
                    data["rmvpe"] = data["f0method"] == "rmvpe"
                    data["fcpe"] = data["f0method"] == "fcpe"
                    
                    # Check if the saved input/output devices exist in the current system
                    if data["sg_input_device"] not in input_devices:
                        data["sg_input_device"] = input_devices[sd.default.device[0]]
                    if data["sg_output_device"] not in output_devices:
                        data["sg_output_device"] = output_devices[sd.default.device[1]]
            except:
                # If the config file does not exist or cannot be read, create a new default config
                data = {
                    "pth_path": "",
                    "index_path": "",
                    "sg_input_device": input_devices[sd.default.device[0]],
                    "sg_output_device": output_devices[sd.default.device[1]],
                    "sr_type": "sr_model",
                    "threshold": -60,
                    "pitch": 0,
                    "index_rate": 0,
                    "rms_mix_rate": 0,
                    "block_time": 0.25,
                    "crossfade_length": 0.05,
                    "extra_time": 2.5,
                    "f0method": "rmvpe",
                    "use_jit": False,
                    "use_pv": False,
                }
                data["sr_model"] = data["sr_type"] == "sr_model"
                data["sr_device"] = data["sr_type"] == "sr_device"
                data["pm"] = data["f0method"] == "pm"
                data["harvest"] = data["f0method"] == "harvest"
                data["crepe"] = data["f0method"] == "crepe"
                data["rmvpe"] = data["f0method"] == "rmvpe"
                data["fcpe"] = data["f0method"] == "fcpe"

            # Load the recent model-index pairs from the history
            recent_model_options = [
                f"Model: {pair['pth_path']} | Index: {pair['index_path']}"
                for pair in self.recent_pairs
            ]

            return data, recent_model_options

        def launch(self):
            # Unpack the returned values from load_config
            data, recent_model_options = self.load_config()

            sg.theme("DarkBlue1")
            input_devices, output_devices, _, _ = self.get_devices()
            layout = [
                [
                    sg.Frame(
                        title=i18n("Load Model"),
                        layout=[
                            [
                                sg.Input(
                                    default_text=data.get("pth_path", ""),
                                    key="pth_path",
                                ),
                                sg.FileBrowse(
                                    i18n("Select .pth file"),
                                    initial_folder=os.path.join(
                                        os.getcwd(), "assets/weights"
                                    ),
                                    file_types=(("PTH Files", "*.pth"),),  # Correct file type format
                                ),
                            ],
                            [
                                sg.Input(
                                    default_text=data.get("index_path", ""),
                                    key="index_path",
                                ),
                                sg.FileBrowse(
                                    i18n("Select .index file"),
                                    initial_folder=os.path.join(os.getcwd(), "logs"),
                                    file_types=(("Index Files", "*.index"),),  # Correct file type format
                                ),
                            ],
                            [
                                sg.Text(i18n("Recently Used")),
                                sg.Combo(
                                    recent_model_options,  # This shows the recent pairs
                                    key="recent_pairs",
                                    enable_events=True,
                                ),
                            ],
                        ],
                    )
                ],
                [
                    sg.Frame(
                        layout=[
                            [
                                sg.Text(i18n("Input Device")),
                                sg.Combo(
                                    input_devices,
                                    key="sg_input_device",
                                    default_value=data.get("sg_input_device", ""),
                                ),
                            ],
                            [
                                sg.Text(i18n("Output Device")),
                                sg.Combo(
                                    output_devices,
                                    key="sg_output_device",
                                    default_value=data.get("sg_output_device", ""),
                                ),
                            ],
                            [
                                sg.Button(i18n("Reload Device List"), key="reload_devices"),
                                sg.Radio(
                                    i18n("Use Model Sample Rate"),
                                    "sr_type",
                                    key="sr_model",
                                    default=data.get("sr_model", True),
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    i18n("Use Device Sample Rate"),
                                    "sr_type",
                                    key="sr_device",
                                    default=data.get("sr_device", False),
                                    enable_events=True,
                                ),
                                sg.Text(i18n("Sample Rate:")),
                                sg.Text("", key="sr_stream"),
                            ],
                        ],
                        title=i18n("Audio Devices (Please use the same type of driver)"),
                    )
                ],
                [
                    sg.Frame(
                        layout=[
                            [
                                sg.Text(i18n("Threshold")),
                                sg.Slider(
                                    range=(-60, 0),
                                    key="threshold",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get("threshold", -60),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("Pitch")),
                                sg.Slider(
                                    range=(-24, 24),
                                    key="pitch",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get("pitch", 0),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("Index Rate")),
                                sg.Slider(
                                    range=(0.0, 1.0),
                                    key="index_rate",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("index_rate", 0),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("Volume Factor")),
                                sg.Slider(
                                    range=(0.0, 1.0),
                                    key="rms_mix_rate",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("rms_mix_rate", 0),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("Pitch Algorithm")),
                                sg.Radio(
                                    "pm",
                                    "f0method",
                                    key="pm",
                                    default=data.get("pm", False),
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    "harvest",
                                    "f0method",
                                    key="harvest",
                                    default=data.get("harvest", False),
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    "crepe",
                                    "f0method",
                                    key="crepe",
                                    default=data.get("crepe", False),
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    "rmvpe",
                                    "f0method",
                                    key="rmvpe",
                                    default=data.get("rmvpe", False),
                                    enable_events=True,
                                ),
                                sg.Radio(
                                    "fcpe",
                                    "f0method",
                                    key="fcpe",
                                    default=data.get("fcpe", True),
                                    enable_events=True,
                                ),
                            ],
                        ],
                        title=i18n("General Settings"),
                    ),
                    sg.Frame(
                        layout=[
                            [
                                sg.Text(i18n("Block Time")),
                                sg.Slider(
                                    range=(0.02, 2.4),
                                    key="block_time",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("block_time", 0.25),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("Number of Harvest Processes")),
                                sg.Slider(
                                    range=(1, n_cpu),
                                    key="n_cpu",
                                    resolution=1,
                                    orientation="h",
                                    default_value=data.get(
                                        "n_cpu", min(self.gui_config.n_cpu, n_cpu)
                                    ),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("Crossfade Length")),
                                sg.Slider(
                                    range=(0.01, 0.15),
                                    key="crossfade_length",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("crossfade_length", 0.05),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Text(i18n("Extra Processing Time")),
                                sg.Slider(
                                    range=(0.05, 5.00),
                                    key="extra_time",
                                    resolution=0.01,
                                    orientation="h",
                                    default_value=data.get("extra_time", 2.5),
                                    enable_events=True,
                                ),
                            ],
                            [
                                sg.Checkbox(
                                    i18n("Input Noise Reduction"),
                                    key="I_noise_reduce",
                                    enable_events=True,
                                ),
                                sg.Checkbox(
                                    i18n("Output Noise Reduction"),
                                    key="O_noise_reduce",
                                    enable_events=True,
                                ),
                                sg.Checkbox(
                                    i18n("Enable Phase Vocoder"),
                                    key="use_pv",
                                    default=data.get("use_pv", False),
                                    enable_events=True,
                                ),
                            ],
                        ],
                        title=i18n("Performance Settings"),
                    ),
                ],
                [
                    sg.Button(i18n("Start Voice Conversion"), key="start_vc"),
                    sg.Button(i18n("Stop Voice Conversion"), key="stop_vc"),
                    sg.Radio(
                        i18n("Input Monitoring"),
                        "function",
                        key="im",
                        default=False,
                        enable_events=True,
                    ),
                    sg.Radio(
                        i18n("Voice Conversion"),
                        "function",
                        key="vc",
                        default=True,
                        enable_events=True,
                    ),
                    sg.Text(i18n("Algorithm Latency (ms):")),
                    sg.Text("0", key="delay_time"),
                    sg.Text(i18n("Inference Time (ms):")),
                    sg.Text("0", key="infer_time"),
                ],
            ]

            self.window = sg.Window("RVC - GUI", layout=layout, finalize=True)
            self.handle_events()



        def handle_events(self):
            while True:
                event, values = self.window.read()
                if event == sg.WINDOW_CLOSED:
                    self.flag_vc = False
                    exit()
                
                # Handle selecting a recent model and index pair from the dropdown
                if event == "recent_pairs":
                    selected = values["recent_pairs"]
                    if selected:
                        # Find the corresponding pair in self.recent_pairs
                        for pair in self.recent_pairs:
                            model_index_str = f"Model: {pair['pth_path']} | Index: {pair['index_path']}"
                            if model_index_str == selected:
                                # Update the input fields with the selected paths
                                self.window["pth_path"].update(pair['pth_path'])
                                self.window["index_path"].update(pair['index_path'])
                                break
                
                # Start voice conversion when the button is pressed
                if event == "start_vc" and not self.flag_vc:
                    if self.set_values(values):
                        self.save_to_history(values["pth_path"], values["index_path"])  # Save the current pair to history
                        self.start_vc()

                        settings = {
                            "pth_path": values["pth_path"],
                            "index_path": values["index_path"],
                            "sg_input_device": values["sg_input_device"],
                            "sg_output_device": values["sg_output_device"],
                            "sr_type": ["sr_model", "sr_device"][
                                [
                                    values["sr_model"],
                                    values["sr_device"],
                                ].index(True)
                            ],
                            "threshold": values["threshold"],
                            "pitch": values["pitch"],
                            "rms_mix_rate": values["rms_mix_rate"],
                            "index_rate": values["index_rate"],
                            "block_time": values["block_time"],
                            "crossfade_length": values["crossfade_length"],
                            "extra_time": values["extra_time"],
                            "n_cpu": values["n_cpu"],
                            "use_jit": False,
                            "use_pv": values["use_pv"],
                            "f0method": ["pm", "harvest", "crepe", "rmvpe", "fcpe"][
                                [
                                    values["pm"],
                                    values["harvest"],
                                    values["crepe"],
                                    values["rmvpe"],
                                    values["fcpe"],
                                ].index(True)
                            ],
                        }
                        with open("configs/config.json", "w") as j:
                            json.dump(settings, j)
                        
                        global stream_latency
                        while stream_latency < 0:
                            time.sleep(0.01)
                        self.delay_time = (
                            stream_latency
                            + values["block_time"]
                            + values["crossfade_length"]
                            + 0.01
                        )
                        if values["I_noise_reduce"]:
                            self.delay_time += min(values["crossfade_length"], 0.04)
                        self.window["sr_stream"].update(self.gui_config.samplerate)
                        self.window["delay_time"].update(int(self.delay_time * 1000))
                
                # Stop voice conversion
                if event == "stop_vc" and self.flag_vc:
                    self.flag_vc = False
                    stream_latency = -1
                
                # Handle parameter updates in real-time
                if event == "threshold":
                    self.gui_config.threshold = values["threshold"]
                elif event == "pitch":
                    self.gui_config.pitch = values["pitch"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_key(values["pitch"])
                elif event == "index_rate":
                    self.gui_config.index_rate = values["index_rate"]
                    if hasattr(self, "rvc"):
                        self.rvc.change_index_rate(values["index_rate"])
                elif event == "rms_mix_rate":
                    self.gui_config.rms_mix_rate = values["rms_mix_rate"]
                elif event in ["pm", "harvest", "crepe", "rmvpe", "fcpe"]:
                    self.gui_config.f0method = event
                elif event == "I_noise_reduce":
                    self.gui_config.I_noise_reduce = values["I_noise_reduce"]
                    if stream_latency > 0:
                        self.delay_time += (
                            1 if values["I_noise_reduce"] else -1
                        ) * min(values["crossfade_length"], 0.04)
                        self.window["delay_time"].update(int(self.delay_time * 1000))
                elif event == "O_noise_reduce":
                    self.gui_config.O_noise_reduce = values["O_noise_reduce"]
                elif event == "use_pv":
                    self.gui_config.use_pv = values["use_pv"]
                elif event in ["vc", "im"]:
                    self.function = event
                elif event != "start_vc" and self.flag_vc:
                    self.flag_vc = False
                    stream_latency = -1
                    
                    
                    
                    

        def set_values(self, values):
            if len(values["pth_path"].strip()) == 0:
                sg.popup(i18n("Please select the .pth file"))
                return False
            if len(values["index_path"].strip()) == 0:
                sg.popup(i18n("Please select the .index file"))
                return False
            pattern = re.compile("[^\x00-\x7F]+")
            if pattern.findall(values["pth_path"]):
                sg.popup(i18n("The .pth file path cannot contain non-ASCII characters"))
                return False
            if pattern.findall(values["index_path"]):
                sg.popup(i18n("The .index file path cannot contain non-ASCII characters"))
                return False
            self.set_devices(values["sg_input_device"], values["sg_output_device"])
            self.config.use_jit = False
            self.gui_config.pth_path = values["pth_path"]
            self.gui_config.index_path = values["index_path"]
            self.gui_config.sr_type = ["sr_model", "sr_device"][
                [
                    values["sr_model"],
                    values["sr_device"],
                ].index(True)
            ]
            self.gui_config.threshold = values["threshold"]
            self.gui_config.pitch = values["pitch"]
            self.gui_config.block_time = values["block_time"]
            self.gui_config.crossfade_time = values["crossfade_length"]
            self.gui_config.extra_time = values["extra_time"]
            self.gui_config.I_noise_reduce = values["I_noise_reduce"]
            self.gui_config.O_noise_reduce = values["O_noise_reduce"]
            self.gui_config.use_pv = values["use_pv"]
            self.gui_config.rms_mix_rate = values["rms_mix_rate"]
            self.gui_config.index_rate = values["index_rate"]
            self.gui_config.n_cpu = values["n_cpu"]
            self.gui_config.f0method = ["pm", "harvest", "crepe", "rmvpe", "fcpe"][
                [
                    values["pm"],
                    values["harvest"],
                    values["crepe"],
                    values["rmvpe"],
                    values["fcpe"],
                ].index(True)
            ]
            return True

        def start_vc(self):
            torch.cuda.empty_cache()
            self.flag_vc = True
            self.rvc = rvc_for_realtime.RVC(
                self.gui_config.pitch,
                self.gui_config.pth_path,
                self.gui_config.index_path,
                self.gui_config.index_rate,
                self.gui_config.n_cpu,
                input_queue,
                output_queue,
                self.config,
                self.rvc if hasattr(self, "rvc") else None,
            )
            self.gui_config.samplerate = self.rvc.tgt_sr if self.gui_config.sr_type == "sr_model" else self.get_device_samplerate()
            self.zc = self.gui_config.samplerate // 100
            self.block_frame = (
                int(
                    np.round(
                        self.gui_config.block_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            self.block_frame_16k = 160 * self.block_frame // self.zc
            self.crossfade_frame = (
                int(
                    np.round(
                        self.gui_config.crossfade_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
            self.sola_search_frame = self.zc
            self.extra_frame = (
                int(
                    np.round(
                        self.gui_config.extra_time
                        * self.gui_config.samplerate
                        / self.zc
                    )
                )
                * self.zc
            )
            self.input_wav: torch.Tensor = torch.zeros(
                self.extra_frame
                + self.crossfade_frame
                + self.sola_search_frame
                + self.block_frame,
                device=self.config.device,
                dtype=torch.float32,
            )
            self.input_wav_res: torch.Tensor = torch.zeros(
                160 * self.input_wav.shape[0] // self.zc,
                device=self.config.device,
                dtype=torch.float32,
            )
            self.sola_buffer: torch.Tensor = torch.zeros(
                self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
            )
            self.nr_buffer: torch.Tensor = self.sola_buffer.clone()
            self.output_buffer: torch.Tensor = self.input_wav.clone()
            self.res_buffer: torch.Tensor = torch.zeros(
                2 * self.zc, device=self.config.device, dtype=torch.float32
            )
            self.skip_head = self.extra_frame // self.zc
            self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc
            self.fade_in_window: torch.Tensor = (
                torch.sin(
                    0.5
                    * np.pi
                    * torch.linspace(
                        0.0,
                        1.0,
                        steps=self.sola_buffer_frame,
                        device=self.config.device,
                        dtype=torch.float32,
                    )
                )
                ** 2
            )
            self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
            self.resampler = tat.Resample(
                orig_freq=self.gui_config.samplerate,
                new_freq=16000,
                dtype=torch.float32,
            ).to(self.config.device)
            if self.rvc.tgt_sr != self.gui_config.samplerate:
                self.resampler2 = tat.Resample(
                    orig_freq=self.rvc.tgt_sr,
                    new_freq=self.gui_config.samplerate,
                    dtype=torch.float32,
                ).to(self.config.device)
            else:
                self.resampler2 = None
            self.tg = TorchGate(
                sr=self.gui_config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
            ).to(self.config.device)
            thread_vc = threading.Thread(target=self.sound_input)
            thread_vc.start()

        def sound_input(self):
            """
            Receive audio input
            """
            channels = 1 if sys.platform == "darwin" else 2
            with sd.Stream(
                channels=channels,
                callback=self.audio_callback,
                blocksize=self.block_frame,
                samplerate=self.gui_config.samplerate,
                dtype="float32",
            ) as stream:
                global stream_latency
                stream_latency = stream.latency[-1]
                while self.flag_vc:
                    time.sleep(self.gui_config.block_time)
                    print_message("Audio block processed.")
            print_message("Ending VC")

        def audio_callback(
            self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
        ):
            """
            Audio processing
            """
            start_time = time.perf_counter()
            indata = librosa.to_mono(indata.T)
            if self.gui_config.threshold > -60:
                rms = librosa.feature.rms(
                    y=indata, frame_length=4 * self.zc, hop_length=self.zc
                )
                db_threshold = (
                    librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threshold
                )
                for i in range(db_threshold.shape[0]):
                    if db_threshold[i]:
                        indata[i * self.zc : (i + 1) * self.zc] = 0
            self.input_wav[: -self.block_frame] = self.input_wav[ self.block_frame :].clone()
            self.input_wav[-self.block_frame :] = torch.from_numpy(indata).to(
                self.config.device
            )
            self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[ self.block_frame_16k :].clone()
            # Input noise reduction and resampling
            if self.gui_config.I_noise_reduce and self.function == "vc":
                input_wav = self.input_wav[ -self.sola_buffer_frame - self.block_frame - 2 * self.zc :]
                input_wav = self.tg(input_wav.unsqueeze(0), self.input_wav.unsqueeze(0))[0, 2 * self.zc :]
                input_wav[: self.sola_buffer_frame] *= self.fade_in_window
                input_wav[: self.sola_buffer_frame] += (self.nr_buffer * self.fade_out_window)
                self.nr_buffer[:] = input_wav[self.block_frame :]
                input_wav = torch.cat((self.res_buffer[:], input_wav[: self.block_frame]))
                self.res_buffer[:] = input_wav[-2 * self.zc :]
                self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(input_wav)[160:]
            else:
                self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(self.input_wav[-self.block_frame - 2 * self.zc :])[160:]
            # Inference
            if self.function == "vc":
                infer_wav = self.rvc.infer(
                    self.input_wav_res,
                    self.block_frame_16k,
                    self.skip_head,
                    self.return_length,
                    self.gui_config.f0method,
                )
                if self.resampler2 is not None:
                    infer_wav = self.resampler2(infer_wav)
            else:
                infer_wav = self.input_wav[-self.crossfade_frame - self.sola_search_frame - self.block_frame :].clone()
            # Output noise reduction
            if (self.gui_config.O_noise_reduce and self.function == "vc") or (self.gui_config.I_noise_reduce and self.function == "im"):
                self.output_buffer[: -self.block_frame] = self.output_buffer[ self.block_frame :].clone()
                self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
                infer_wav = self.tg(infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)
            # Volume envelope mixing
            if self.gui_config.rms_mix_rate < 1 and self.function == "vc":
                rms1 = librosa.feature.rms(y=self.input_wav_res[160 * self.skip_head : 160 * (self.skip_head + self.return_length)].cpu().numpy(), frame_length=640, hop_length=160,)
                rms1 = torch.from_numpy(rms1).to(self.config.device)
                rms1 = F.interpolate(rms1.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True,)[0, 0, :-1]
                rms2 = librosa.feature.rms(y=infer_wav[:].cpu().numpy(), frame_length=4 * self.zc, hop_length=self.zc,)
                rms2 = torch.from_numpy(rms2).to(self.config.device)
                rms2 = F.interpolate(rms2.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True,)[0, 0, :-1]
                rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
                infer_wav *= torch.pow(rms1 / rms2, torch.tensor(1 - self.gui_config.rms_mix_rate))
            # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
            conv_input = infer_wav[ None, None, : self.sola_buffer_frame + self.sola_search_frame ]
            cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(F.conv1d(conv_input**2, torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),) + 1e-8)
            if sys.platform == "darwin":
                _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
                sola_offset = sola_offset.item()
            else:
                sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
            print_message("SOLA offset = %d", int(sola_offset))
            infer_wav = infer_wav[sola_offset :]
            if "privateuseone" in str(self.config.device) or not self.gui_config.use_pv:
                infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
                infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
            else:
                infer_wav[: self.sola_buffer_frame] = phase_vocoder(self.sola_buffer, infer_wav[: self.sola_buffer_frame], self.fade_out_window, self.fade_in_window)
            self.sola_buffer[:] = infer_wav[self.block_frame : self.block_frame + self.sola_buffer_frame]
            if sys.platform == "darwin":
                outdata[:] = infer_wav[: self.block_frame].cpu().numpy()[:, np.newaxis]
            else:
                outdata[:] = infer_wav[: self.block_frame].repeat(2, 1).t().cpu().numpy()
            total_time = time.perf_counter() - start_time
            self.window["infer_time"].update(int(total_time * 1000))
            print_message("Inference time: %.2f", total_time)

        def get_devices(self, update: bool = True):
            """Get the list of devices"""
            if update:
                sd._terminate()
                sd._initialize()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            for hostapi in hostapis:
                for device_idx in hostapi["devices"]:
                    devices[device_idx]["hostapi_name"] = hostapi["name"]
            input_devices = [
                f"{d['name']} ({d['hostapi_name']})"
                for d in devices
                if d["max_input_channels"] > 0
            ]
            output_devices = [
                f"{d['name']} ({d['hostapi_name']})"
                for d in devices
                if d["max_output_channels"] > 0
            ]
            input_device_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_input_channels"] > 0
            ]
            output_device_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_output_channels"] > 0
            ]
            return (
                input_devices,
                output_devices,
                input_device_indices,
                output_device_indices,
            )
                    
        def set_devices(self, input_device, output_device):
            """Set the output devices"""
            input_devices, output_devices, input_device_indices, output_device_indices = self.get_devices()
            sd.default.device[0] = input_device_indices[input_devices.index(input_device)]
            sd.default.device[1] = output_device_indices[output_devices.index(output_device)]
            print_message("Input device: %s:%s", str(sd.default.device[0]), input_device)
            print_message("Output device: %s:%s", str(sd.default.device[1]), output_device)
        
        def get_device_samplerate(self):
            return int(sd.query_devices(device=sd.default.device[0])['default_samplerate'])

    gui = GUI()
