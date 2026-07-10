from __future__ import annotations

import logging
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from autodj.audio_engine import AutoDJEngine, EngineConfig
from autodj.allinone_analyzer import AllInOneAnalyzer, probe_allinone
from autodj.beat_this_analyzer import BeatThisAnalyzer
from autodj.cuedetr_analyzer import CueDETRAnalyzer, probe_cuedetr
from autodj.models import AllInOneProfile, CueDETRProfile, MuQProfile, TrackAnalysis
from autodj.muq_analyzer import MuQAnalyzer
from autodj.playlist_ranker import PairScore, rank_playlist, style_clusters, transition_compatibility
from autodj.timeline import DJTimeline
from autodj.time_stretch import rubberband_probe
from autodj.settings_store import SettingsStore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("autodj")


AUDIO_FILETYPES = [
    ("音频文件", "*.wav *.flac *.ogg *.aiff *.aif *.mp3 *.m4a"),
    ("WAV", "*.wav"),
    ("FLAC", "*.flac"),
    ("所有文件", "*.*"),
]


def format_time(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    value = max(0, int(seconds))
    return f"{value // 60:02d}:{value % 60:02d}"


class VerticalScrolledFrame(ttk.Frame):
    """带独立垂直滚动条的 ttk 容器。

    鼠标位于面板上方时支持滚轮；内部宽度始终跟随可视区域，
    因此标签和下拉框不会被裁到窗口外。
    """

    def __init__(self, master: tk.Misc, **kwargs: object) -> None:
        super().__init__(master, style="Card.TFrame", **kwargs)
        self.canvas = tk.Canvas(
            self,
            bg="#202328",
            highlightthickness=0,
            borderwidth=0,
            relief=tk.FLAT,
        )
        self.scrollbar = ttk.Scrollbar(
            self, orient=tk.VERTICAL, command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.content = ttk.Frame(self.canvas, style="Card.TFrame")
        self._window_id = self.canvas.create_window(
            (0, 0), window=self.content, anchor="nw"
        )

        self.content.bind("<Configure>", self._sync_scroll_region)
        self.canvas.bind("<Configure>", self._sync_content_width)
        # 全局接收滚轮，但仅当鼠标确实位于本面板内部时处理。
        # 这样在 Label、Combobox、Scale 等子控件上方也能滚动，
        # 且不会破坏播放列表或其他组件自己的滚轮行为。
        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.bind_all("<Button-4>", self._on_linux_wheel, add="+")
        self.bind_all("<Button-5>", self._on_linux_wheel, add="+")

    def _sync_scroll_region(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _sync_content_width(self, event: tk.Event[tk.Misc]) -> None:
        self.canvas.itemconfigure(self._window_id, width=max(1, event.width))

    def _pointer_is_inside(self) -> bool:
        try:
            x, y = self.winfo_pointerxy()
            widget = self.winfo_containing(x, y)
        except tk.TclError:
            return False
        while widget is not None:
            if widget is self:
                return True
            widget = getattr(widget, "master", None)
        return False

    def _on_mousewheel(self, event: tk.Event[tk.Misc]) -> str | None:
        if not self._pointer_is_inside():
            return None
        delta = int(getattr(event, "delta", 0))
        if delta:
            steps = -1 if delta > 0 else 1
            if abs(delta) >= 120:
                steps = -int(delta / 120)
            self.canvas.yview_scroll(steps, "units")
        return "break"

    def _on_linux_wheel(self, event: tk.Event[tk.Misc]) -> str | None:
        if not self._pointer_is_inside():
            return None
        number = int(getattr(event, "num", 0))
        self.canvas.yview_scroll(-1 if number == 4 else 1, "units")
        return "break"

    def scroll_to_top(self) -> None:
        self.canvas.yview_moveto(0.0)


class AutoDJApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Beat This! + CUE-DETR + MuQ + All-In-One Auto DJ 1.2.13")
        self.settings_store = SettingsStore()
        self.saved_settings = self.settings_store.load()
        self._settings_after_id: str | None = None

        # 根据实际屏幕尺寸决定首次窗口大小；若上次正常退出时保存了
        # geometry，则优先恢复用户布局。
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = min(1240, max(760, screen_width - 80))
        window_height = min(860, max(560, screen_height - 100))
        window_x = max(0, (screen_width - window_width) // 2)
        window_y = max(0, min(30, (screen_height - window_height) // 3))
        default_geometry = f"{window_width}x{window_height}+{window_x}+{window_y}"
        geometry = str(self.saved_settings.get("window_geometry", default_geometry))
        try:
            self.geometry(geometry)
        except tk.TclError:
            self.geometry(default_geometry)
        self.minsize(
            min(820, max(680, screen_width - 80)),
            min(600, max(520, screen_height - 80)),
        )

        self.engine = AutoDJEngine(EngineConfig())
        self.tracks: list[TrackAnalysis] = []
        self.ui_events: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self.analysis_active = False
        self.starting_playback = False
        self.device_map: dict[str, int | None] = {"系统默认": None}
        self.last_beat_number = -1
        self.last_engine_index = -1
        self.last_plan_signature: object = object()
        self.last_preload_signature: object = object()
        self.muq_profiles: dict[str, MuQProfile] = {}
        self.muq_groups: dict[str, int] = {}
        self.muq_pair_scores: dict[str, PairScore] = {}
        self.muq_ranking_active = False
        self.allin1_profiles: dict[str, AllInOneProfile] = {}
        self.allin1_analysis_active = False
        self.allin1_probe_result: dict[str, object] = {}
        self.cuedetr_profiles: dict[str, CueDETRProfile] = {}
        self.cuedetr_analysis_active = False
        self.cuedetr_probe_result: dict[str, object] = {}
        self._preload_after_id: str | None = None

        self._build_style()
        self._build_ui()
        self._restore_saved_settings()
        self._refresh_devices()
        self._attach_settings_autosave()
        self._apply_engine_settings()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Configure>", lambda _event: self._schedule_settings_save(), add="+")
        self.after(150, self._detect_rubberband)
        self.after(240, self._detect_allin1)
        self.after(320, self._detect_cuedetr)
        LOGGER.info("Auto DJ 1.2.13 GUI 启动，主环境 Python=%s", os.sys.executable)
        self.after(100, self._poll)

    _SETTING_VARIABLES = {
        "auto_mix": "auto_mix_var",
        "volume": "volume_var",
        "transition_bars": "bars_var",
        "mix_style": "style_var",
        "effect_strength": "effect_var",
        "human_style": "human_style_var",
        "human_candidates": "human_candidates_var",
        "tempo_restore": "restore_var",
        "automix_policy": "policy_var",
        "transition_engine": "transition_engine_var",
        "stretch_backend": "stretch_backend_var",
        "rubberband_path": "rb_path_var",
        "max_stretch": "stretch_var",
        "allin1_enabled": "allin1_enabled_var",
        "allin1_model": "allin1_model_var",
        "allin1_device": "allin1_device_var",
        "cuedetr_model": "cuedetr_model_var",
        "cuedetr_device": "cuedetr_device_var",
        "cuedetr_sensitivity": "cuedetr_sensitivity_var",
        "cuedetr_min_bars": "cuedetr_min_bars_var",
        "muq_enabled": "muq_enabled_var",
        "auto_muq_sort": "auto_muq_sort_var",
        "preload_enabled": "preload_pair_var",
        "preload_window": "preload_window_var",
        "preload_memory_mb": "preload_memory_var",
        "preload_deadline_seconds": "preload_deadline_var",
        "beat_this_model": "model_var",
        "compute_device": "compute_var",
        "output_device": "output_var",
    }

    def _restore_saved_settings(self) -> None:
        # v1.2.2 defaulted the shared compute device to CPU. Treat that legacy
        # default as automatic once; users can still explicitly choose CPU later.
        legacy_device = str(self.saved_settings.get("compute_device", "")).lower()
        if legacy_device == "cpu" and not self.saved_settings.get("device_default_migrated_v123"):
            self.saved_settings["compute_device"] = "auto"
            self.saved_settings["device_default_migrated_v123"] = True
        for key, attribute in self._SETTING_VARIABLES.items():
            variable = getattr(self, attribute, None)
            if variable is None or key not in self.saved_settings:
                continue
            try:
                variable.set(self.saved_settings[key])
            except (tk.TclError, TypeError, ValueError):
                continue
        # Refresh text next to scales immediately.
        if self.human_style_var.get() == "Adaptive Human":
            self.human_style_var.set("Natural Auto")
        elif self.human_style_var.get() == "Long Blend":
            self.human_style_var.set("Short Blend")
        self.effect_label.configure(text=f"{float(self.effect_var.get()):.0f}%")
        self.stretch_label.configure(text=f"±{float(self.stretch_var.get()):.1f}%")

    def _attach_settings_autosave(self) -> None:
        for attribute in self._SETTING_VARIABLES.values():
            variable = getattr(self, attribute, None)
            if variable is not None:
                variable.trace_add(
                    "write", lambda *_args: self._settings_changed()
                )

    def _settings_changed(self) -> None:
        # Existing widgets keep their focused engine callbacks. Only the new
        # sliding-window values need a generic trace-based apply step; avoiding a
        # full _apply_engine_settings() here prevents expensive transition rebuilds
        # while the user drags the volume/effect sliders.
        try:
            if hasattr(self, "preload_window_var"):
                self.engine.set_preload_window_tracks(int(self.preload_window_var.get()))
                self.engine.set_preload_memory_mb(int(self.preload_memory_var.get()))
                self.engine.set_preload_deadline_seconds(
                    float(self.preload_deadline_var.get())
                )
        except (ValueError, tk.TclError):
            pass
        self._schedule_settings_save()
        if hasattr(self, "settings_saved_label"):
            self.settings_saved_label.configure(text="配置有变更，正在保存…")

    def _collect_settings(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for key, attribute in self._SETTING_VARIABLES.items():
            variable = getattr(self, attribute, None)
            if variable is not None:
                values[key] = variable.get()
        if self.saved_settings.get("device_default_migrated_v123"):
            values["device_default_migrated_v123"] = True
        try:
            values["window_geometry"] = self.geometry()
        except tk.TclError:
            pass
        return values

    def _schedule_settings_save(self, delay_ms: int = 450) -> None:
        if self._settings_after_id is not None:
            try:
                self.after_cancel(self._settings_after_id)
            except tk.TclError:
                pass
        self._settings_after_id = self.after(delay_ms, self._save_settings_now)

    def _save_settings_now(self) -> None:
        self._settings_after_id = None
        try:
            self.settings_store.save(self._collect_settings())
            if hasattr(self, "settings_saved_label"):
                self.settings_saved_label.configure(
                    text=f"配置已自动保存：{self.settings_store.path}"
                )
        except Exception as exc:
            if hasattr(self, "settings_saved_label"):
                self.settings_saved_label.configure(text=f"配置保存失败：{exc}")

    def _build_style(self) -> None:
        self.configure(bg="#15171a")
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background="#15171a", foreground="#f1f3f5")
        style.configure("TFrame", background="#15171a")
        style.configure("Card.TFrame", background="#202328")
        style.configure(
            "TLabel",
            background="#15171a",
            foreground="#f1f3f5",
            font=("Arial", 10),
        )
        style.configure(
            "Title.TLabel",
            font=("Arial", 21, "bold"),
            foreground="#ffffff",
        )
        style.configure(
            "Now.TLabel",
            background="#202328",
            foreground="#ffffff",
            font=("Arial", 17, "bold"),
        )
        style.configure(
            "Muted.TLabel",
            background="#202328",
            foreground="#adb5bd",
        )
        style.configure(
            "Metric.TLabel",
            background="#202328",
            foreground="#66d9ef",
            font=("Arial", 10, "bold"),
        )
        style.configure("TButton", font=("Arial", 10, "bold"), padding=8)
        style.configure("Accent.TButton", background="#ff4d6d", foreground="#ffffff")
        style.map(
            "Accent.TButton",
            background=[("active", "#ff758f"), ("disabled", "#6c454c")],
        )
        style.configure(
            "Treeview",
            background="#202328",
            foreground="#f1f3f5",
            fieldbackground="#202328",
            rowheight=29,
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background="#2b2f35",
            foreground="#ffffff",
            font=("Arial", 10, "bold"),
        )
        style.map("Treeview", background=[("selected", "#495057")])
        style.configure("TCheckbutton", background="#15171a", foreground="#f1f3f5")
        style.configure(
            "TCombobox",
            fieldbackground="#2b2f35",
            background="#2b2f35",
            foreground="#ffffff",
        )

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X)
        header.grid_columnconfigure(1, weight=1)
        ttk.Label(header, text="Beat This! + CUE-DETR + MuQ + All-In-One Auto DJ 1.2.13", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.header_subtitle = ttk.Label(
            header,
            text="MuQ 平滑排序 · 自由 OUT/IN 乐句配对 · 分级滑动预渲染 · 配置自动保存",
            justify=tk.LEFT,
        )
        self.header_subtitle.grid(
            row=0, column=1, sticky="ew", padx=(18, 0), pady=(8, 0)
        )
        header.bind(
            "<Configure>",
            lambda event: self.header_subtitle.configure(
                wraplength=max(180, event.width - 330)
            ),
        )

        now_card = ttk.Frame(outer, style="Card.TFrame", padding=12)
        now_card.pack(fill=tk.X, pady=(10, 8))
        now_card.grid_columnconfigure(0, weight=1)
        now_left = ttk.Frame(now_card, style="Card.TFrame")
        now_left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        ttk.Label(now_left, text="NOW PLAYING", style="Muted.TLabel").pack(anchor=tk.W)
        self.now_label = ttk.Label(
            now_left, text="尚未播放", style="Now.TLabel", justify=tk.LEFT
        )
        self.now_label.pack(anchor=tk.W, fill=tk.X, pady=(3, 2))
        self.next_label = ttk.Label(
            now_left, text="下一首：—", style="Muted.TLabel", justify=tk.LEFT
        )
        self.next_label.pack(anchor=tk.W, fill=tk.X)
        self.plan_label = ttk.Label(
            now_left,
            text="切歌规划：等待下一首分析",
            style="Metric.TLabel",
            justify=tk.LEFT,
        )
        self.plan_label.pack(anchor=tk.W, fill=tk.X, pady=(5, 0))

        now_right = ttk.Frame(now_card, style="Card.TFrame")
        now_right.grid(row=0, column=1, sticky="ne")
        now_card.bind(
            "<Configure>",
            lambda event: [
                widget.configure(wraplength=max(220, event.width - 360))
                for widget in (self.now_label, self.next_label, self.plan_label)
            ],
        )
        self.beat_label = tk.Label(
            now_right,
            text="BEAT\n—",
            width=8,
            bg="#343a40",
            fg="#ffffff",
            font=("Arial", 16, "bold"),
            padx=9,
            pady=9,
        )
        self.beat_label.pack(side=tk.LEFT, padx=(10, 0))
        self.bpm_label = tk.Label(
            now_right,
            text="0.0\nBPM",
            width=8,
            bg="#343a40",
            fg="#ffffff",
            font=("Arial", 16, "bold"),
            padx=9,
            pady=9,
        )
        self.bpm_label.pack(side=tk.LEFT, padx=(10, 0))
        self.score_label = tk.Label(
            now_right,
            text="MATCH\n—",
            width=8,
            bg="#343a40",
            fg="#ffffff",
            font=("Arial", 16, "bold"),
            padx=9,
            pady=9,
        )
        self.score_label.pack(side=tk.LEFT, padx=(10, 0))

        timeline_card = ttk.Frame(outer, style="Card.TFrame", padding=8)
        timeline_card.pack(fill=tk.X)
        self.timeline = DJTimeline(timeline_card, on_seek=self._seek_to)
        self.timeline.pack(fill=tk.X, expand=True)
        time_row = ttk.Frame(timeline_card, style="Card.TFrame")
        time_row.pack(fill=tk.X, padx=55, pady=(2, 0))
        self.time_left = ttk.Label(time_row, text="00:00", style="Muted.TLabel")
        self.time_left.pack(side=tk.LEFT)
        ttk.Label(
            time_row,
            text="点击或拖动上方 A 轨跳转（自动吸附节拍）",
            style="Muted.TLabel",
        ).pack(side=tk.LEFT, expand=True)
        self.time_right = ttk.Label(time_row, text="00:00", style="Muted.TLabel")
        self.time_right.pack(side=tk.RIGHT)

        transport = ttk.Frame(outer)
        transport.pack(fill=tk.X, pady=8)
        button_row = ttk.Frame(transport)
        button_row.pack(fill=tk.X)
        self.play_button = ttk.Button(
            button_row,
            text="▶ 播放",
            style="Accent.TButton",
            command=self._play_or_resume,
        )
        self.play_button.pack(side=tk.LEFT)
        ttk.Button(button_row, text="⏸ 暂停", command=self._pause).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(button_row, text="⏹ 停止", command=self._stop).pack(side=tk.LEFT)
        ttk.Button(
            button_row,
            text="⏭ 下个重拍智能切歌",
            command=self._next,
        ).pack(side=tk.LEFT, padx=(12, 0))

        self.auto_mix_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            button_row,
            text="自动切歌",
            variable=self.auto_mix_var,
            command=lambda: self.engine.set_auto_mix(self.auto_mix_var.get()),
        ).pack(side=tk.LEFT, padx=(12, 0))

        volume_row = ttk.Frame(transport)
        volume_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(volume_row, text="主音量").pack(side=tk.LEFT)
        self.volume_var = tk.DoubleVar(value=90)
        ttk.Scale(
            volume_row,
            from_=0,
            to=125,
            variable=self.volume_var,
            command=lambda value: self.engine.set_volume(float(value) / 100.0),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        middle = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        middle.pack(fill=tk.BOTH, expand=True)
        queue_frame = ttk.Frame(middle, style="Card.TFrame", padding=12)
        settings_shell = ttk.Frame(middle, style="Card.TFrame")
        self.settings_scroll = VerticalScrolledFrame(settings_shell)
        self.settings_scroll.pack(fill=tk.BOTH, expand=True)
        settings_frame = ttk.Frame(
            self.settings_scroll.content, style="Card.TFrame", padding=14
        )
        settings_frame.pack(fill=tk.BOTH, expand=True)
        middle.add(queue_frame, weight=5)
        middle.add(settings_shell, weight=2)

        queue_toolbar = ttk.Frame(queue_frame, style="Card.TFrame")
        queue_toolbar.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(queue_toolbar, text="+ 添加歌曲", command=self._add_files).pack(
            side=tk.LEFT
        )
        ttk.Button(queue_toolbar, text="移除", command=self._remove_selected).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(queue_toolbar, text="清空", command=self._clear).pack(side=tk.LEFT)
        ttk.Button(
            queue_toolbar,
            text="重新分析",
            command=self._reanalyze_selected,
        ).pack(side=tk.LEFT, padx=(14, 0))
        ttk.Button(
            queue_toolbar,
            text="All-In-One 结构",
            command=lambda: self._analyze_allin1(force=True, automatic=False),
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            queue_toolbar,
            text="MuQ 智能排序",
            style="Accent.TButton",
            command=self._rank_with_muq,
        ).pack(side=tk.LEFT, padx=(8, 0))

        table_frame = ttk.Frame(queue_frame, style="Card.TFrame")
        table_frame.pack(fill=tk.BOTH, expand=True)
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        columns = ("index", "title", "bpm", "meter", "duration", "structure", "muq", "compat", "mix", "state")
        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
        )
        headings = {
            "index": "#",
            "title": "歌曲",
            "bpm": "BPM",
            "meter": "拍号",
            "duration": "时长",
            "structure": "All-In-One结构",
            "muq": "MuQ风格组",
            "compat": "与下一首",
            "mix": "智能切歌点",
            "state": "状态",
        }
        widths = {
            "index": 38,
            "title": 250,
            "bpm": 65,
            "meter": 55,
            "duration": 65,
            "structure": 140,
            "muq": 78,
            "compat": 86,
            "mix": 110,
            "state": 80,
        }
        for key in columns:
            self.tree.heading(key, text=headings[key])
            self.tree.column(
                key,
                width=widths[key],
                minwidth=38,
                stretch=(key == "title"),
                anchor=tk.W if key == "title" else tk.CENTER,
            )
        vscroll = ttk.Scrollbar(
            table_frame, orient=tk.VERTICAL, command=self.tree.yview
        )
        hscroll = ttk.Scrollbar(
            table_frame, orient=tk.HORIZONTAL, command=self.tree.xview
        )
        self.tree.configure(
            yscrollcommand=vscroll.set, xscrollcommand=hscroll.set
        )
        self.tree.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<Double-1>", lambda _: self._play_or_resume())
        self.tree.bind("<<TreeviewSelect>>", lambda _: self._schedule_preload())

        ttk.Label(settings_frame, text="智能混音设置", style="Now.TLabel").pack(
            anchor=tk.W, pady=(0, 14)
        )
        ttk.Label(
            settings_frame,
            text="Cue 配对上下文",
            style="Muted.TLabel",
        ).pack(anchor=tk.W)
        self.bars_var = tk.StringVar(value="自动")
        bars = ttk.Combobox(
            settings_frame,
            textvariable=self.bars_var,
            values=("自动", "4", "8", "16", "32"),
            state="readonly",
        )
        bars.pack(fill=tk.X, pady=(4, 4))
        bars.bind("<<ComboboxSelected>>", lambda _: self._apply_bars())
        ttk.Label(
            settings_frame,
            text="只影响 cue 前后的结构评分；实际可听过渡固定为 cue 前后各约 1 拍。",
            style="Muted.TLabel",
            wraplength=280,
        ).pack(anchor=tk.W, pady=(0, 12))

        ttk.Label(settings_frame, text="专业混音风格", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        self.style_var = tk.StringVar(value="Club")
        style_box = ttk.Combobox(
            settings_frame,
            textvariable=self.style_var,
            values=("Smooth", "Club", "Filter", "Echo"),
            state="readonly",
        )
        style_box.pack(fill=tk.X, pady=(4, 10))
        style_box.bind("<<ComboboxSelected>>", lambda _: self._apply_style())

        ttk.Label(settings_frame, text="效果强度", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        self.effect_var = tk.DoubleVar(value=72)
        self.effect_label = ttk.Label(settings_frame, text="72%", style="Muted.TLabel")
        self.effect_label.pack(anchor=tk.E)
        ttk.Scale(
            settings_frame,
            from_=0,
            to=100,
            variable=self.effect_var,
            command=self._set_effect_strength,
        ).pack(fill=tk.X, pady=(0, 10))

        ttk.Label(settings_frame, text="真人 DJ 过渡策略", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        self.human_style_var = tk.StringVar(value="Natural Auto")
        human_style_box = ttk.Combobox(
            settings_frame,
            textvariable=self.human_style_var,
            values=(
                "Natural Auto",
                "Short Blend",
                "Bass Swap",
                "Echo Out",
            ),
            state="readonly",
        )
        human_style_box.pack(fill=tk.X, pady=(4, 4))
        human_style_box.bind(
            "<<ComboboxSelected>>", lambda _: self._apply_human_style()
        )
        ttk.Label(
            settings_frame,
            text="Natural Auto 只使用 cue 居中的短融合、低频交接和必要的人声 Echo 退出；切点后下一首立即占主导，但保留极短平滑尾巴避免硬断。",
            style="Muted.TLabel",
            wraplength=280,
        ).pack(anchor=tk.W, pady=(0, 8))

        candidate_row = ttk.Frame(settings_frame, style="Card.TFrame")
        candidate_row.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            candidate_row, text="候选手法数", style="Muted.TLabel"
        ).pack(side=tk.LEFT)
        self.human_candidates_var = tk.StringVar(value="3")
        human_candidates_box = ttk.Combobox(
            candidate_row,
            textvariable=self.human_candidates_var,
            values=("1", "2", "3"),
            state="readonly",
            width=5,
        )
        human_candidates_box.pack(side=tk.RIGHT)
        human_candidates_box.bind(
            "<<ComboboxSelected>>", lambda _: self._apply_human_candidates()
        )

        ttk.Label(settings_frame, text="切歌后 BPM 回归", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        self.restore_var = tk.StringVar(value="自动")
        restore_box = ttk.Combobox(
            settings_frame,
            textvariable=self.restore_var,
            values=("自动", "关闭", "4 小节", "8 小节", "16 小节", "32 小节"),
            state="readonly",
        )
        restore_box.pack(fill=tk.X, pady=(4, 4))
        restore_box.bind("<<ComboboxSelected>>", lambda _: self._apply_restore())
        ttk.Label(
            settings_frame,
            text="混音时先同步 BPM，结束后平滑回到下一首原 BPM",
            style="Muted.TLabel",
            wraplength=280,
        ).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(settings_frame, text="AutoMix 策略", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        self.policy_var = tk.StringVar(value="AutoMix-like")
        policy_box = ttk.Combobox(
            settings_frame,
            textvariable=self.policy_var,
            values=("AutoMix-like", "Always DJ", "Crossfade"),
            state="readonly",
        )
        policy_box.pack(fill=tk.X, pady=(4, 4))
        policy_box.bind("<<ComboboxSelected>>", lambda _: self._apply_policy())
        ttk.Label(
            settings_frame,
            text="按匹配置信度自动选择复杂 DJ 过渡、简单淡化或静音裁切无缝播放",
            style="Muted.TLabel",
            wraplength=280,
        ).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(settings_frame, text="过渡渲染引擎", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        self.transition_engine_var = tk.StringVar(value="Adaptive")
        transition_engine_box = ttk.Combobox(
            settings_frame,
            textvariable=self.transition_engine_var,
            values=("Adaptive", "Spectral Seam", "EQ/Fader"),
            state="readonly",
        )
        transition_engine_box.pack(fill=tk.X, pady=(4, 4))
        transition_engine_box.bind(
            "<<ComboboxSelected>>", lambda _: self._apply_transition_engine()
        )
        ttk.Label(
            settings_frame,
            text="Spectral Seam 使用论文中的时频图最小割缝合；长过渡会自动回退三段 EQ",
            style="Muted.TLabel",
            wraplength=280,
        ).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(settings_frame, text="时间拉伸质量", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        self.stretch_backend_var = tk.StringVar(value="auto")
        stretch_backend_box = ttk.Combobox(
            settings_frame,
            textvariable=self.stretch_backend_var,
            values=("auto", "Rubber Band R3", "Hybrid HPSS", "librosa"),
            state="readonly",
        )
        stretch_backend_box.pack(fill=tk.X, pady=(4, 4))
        stretch_backend_box.bind(
            "<<ComboboxSelected>>", lambda _: self._apply_stretch_backend()
        )
        ttk.Label(
            settings_frame,
            text="auto 会优先调用 Rubber Band R3；未安装时回退 Hybrid HPSS + WSOLA，减少电子鼓瞬态涂抹",
            style="Muted.TLabel",
            wraplength=280,
        ).pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(settings_frame, text="Rubber Band CLI 路径", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        self.rb_path_var = tk.StringVar(
            value=(
                os.environ.get("AUTODJ_RUBBERBAND")
                or os.environ.get("RUBBERBAND_EXE")
                or os.environ.get("RUBBERBAND_PATH")
                or ""
            )
        )
        rb_row = ttk.Frame(settings_frame, style="Card.TFrame")
        rb_row.pack(fill=tk.X, pady=(4, 4))
        self.rb_path_entry = ttk.Entry(rb_row, textvariable=self.rb_path_var)
        self.rb_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            rb_row, text="选择…", width=7, command=self._choose_rubberband
        ).pack(side=tk.LEFT, padx=(6, 0))
        rb_actions = ttk.Frame(settings_frame, style="Card.TFrame")
        rb_actions.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(
            rb_actions, text="应用并检测", command=self._apply_rubberband_path
        ).pack(side=tk.LEFT)
        ttk.Button(
            rb_actions, text="清除手动路径", command=self._clear_rubberband_path
        ).pack(side=tk.LEFT, padx=(6, 0))
        self.rb_status_label = ttk.Label(
            settings_frame,
            text="正在检测 Rubber Band…",
            style="Muted.TLabel",
            wraplength=280,
        )
        self.rb_status_label.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))

        ttk.Label(settings_frame, text="最大变速百分比", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        self.stretch_var = tk.DoubleVar(value=12)
        self.stretch_label = ttk.Label(
            settings_frame,
            text="±12.0%",
            style="Muted.TLabel",
        )
        self.stretch_label.pack(anchor=tk.E)
        ttk.Scale(
            settings_frame,
            from_=0,
            to=25,
            variable=self.stretch_var,
            command=self._set_stretch,
        ).pack(fill=tk.X, pady=(0, 12))

        ttk.Separator(settings_frame).pack(fill=tk.X, pady=10)
        ttk.Label(settings_frame, text="All-In-One 歌曲结构", style="Now.TLabel").pack(
            anchor=tk.W, pady=(0, 6)
        )
        self.allin1_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            settings_frame,
            text="启用 All-In-One 官方功能段模型",
            variable=self.allin1_enabled_var,
            command=lambda: self.engine.set_allin1_enabled(self.allin1_enabled_var.get()),
        ).pack(anchor=tk.W)
        ttk.Label(settings_frame, text="All-In-One 模型", style="Muted.TLabel").pack(
            anchor=tk.W, pady=(6, 0)
        )
        self.allin1_model_var = tk.StringVar(value="harmonix-all")
        allin1_box = ttk.Combobox(
            settings_frame,
            textvariable=self.allin1_model_var,
            values=(
                "harmonix-all",
                "harmonix-fold0",
                "harmonix-fold1",
                "harmonix-fold2",
                "harmonix-fold3",
                "harmonix-fold4",
                "harmonix-fold5",
                "harmonix-fold6",
                "harmonix-fold7",
            ),
            state="readonly",
        )
        allin1_box.pack(fill=tk.X, pady=(4, 4))
        allin1_box.bind(
            "<<ComboboxSelected>>",
            lambda _: self._apply_allin1_model(),
        )
        ttk.Label(settings_frame, text="All-In-One 运行设备", style="Muted.TLabel").pack(
            anchor=tk.W, pady=(6, 0)
        )
        self.allin1_device_var = tk.StringVar(value="cpu")
        ttk.Combobox(
            settings_frame,
            textvariable=self.allin1_device_var,
            values=("cpu", "cuda", "mps"),
            state="readonly",
        ).pack(fill=tk.X, pady=(4, 4))
        self.allin1_status_label = ttk.Label(
            settings_frame,
            text="尚未检测 All-In-One。模型直接在主进程后台线程中运行。",
            style="Muted.TLabel",
            wraplength=280,
        )
        self.allin1_status_label.pack(anchor=tk.W, fill=tk.X, pady=(2, 4))
        allin1_actions = ttk.Frame(settings_frame, style="Card.TFrame")
        allin1_actions.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(
            allin1_actions,
            text="检测模型",
            command=self._detect_allin1,
        ).pack(side=tk.LEFT)
        ttk.Button(
            allin1_actions,
            text="分析全部结构",
            command=lambda: self._analyze_allin1(force=False, automatic=False),
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(
            settings_frame,
            text=(
                "Beat This! 继续负责节拍网格；All-In-One 负责 intro、verse、"
                "chorus、break、bridge、solo、outro。它直接在主进程后台线程运行；"
                "默认使用 CPU，避免占满 RTX 5070 显存。分析结果会缓存。"
            ),
            style="Muted.TLabel",
            wraplength=280,
        ).pack(anchor=tk.W, pady=(0, 10))

        ttk.Separator(settings_frame).pack(fill=tk.X, pady=10)
        ttk.Label(settings_frame, text="CUE-DETR 专业切点", style="Now.TLabel").pack(
            anchor=tk.W, pady=(0, 6)
        )
        self.cuedetr_enabled_var = tk.BooleanVar(value=True)
        ttk.Label(
            settings_frame,
            text="CUE-DETR 固定启用（唯一切点来源）",
            style="Muted.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(settings_frame, text="CUE-DETR 模型", style="Muted.TLabel").pack(
            anchor=tk.W, pady=(6, 0)
        )
        self.cuedetr_model_var = tk.StringVar(value="disco-eth/cue-detr")
        ttk.Entry(settings_frame, textvariable=self.cuedetr_model_var).pack(
            fill=tk.X, pady=(4, 4)
        )
        cue_grid = ttk.Frame(settings_frame, style="Card.TFrame")
        cue_grid.pack(fill=tk.X, pady=(4, 4))
        cue_grid.grid_columnconfigure(1, weight=1)
        ttk.Label(cue_grid, text="运行设备", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=2
        )
        self.cuedetr_device_var = tk.StringVar(value="auto")
        ttk.Combobox(
            cue_grid, textvariable=self.cuedetr_device_var,
            values=("auto", "cuda", "cpu", "mps"), state="readonly", width=10
        ).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(cue_grid, text="灵敏度", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=2
        )
        self.cuedetr_sensitivity_var = tk.StringVar(value="0.90")
        ttk.Combobox(
            cue_grid, textvariable=self.cuedetr_sensitivity_var,
            values=("0.75", "0.82", "0.88", "0.90", "0.93", "0.96"),
            state="readonly", width=10
        ).grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Label(cue_grid, text="最小间隔", style="Muted.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=2
        )
        self.cuedetr_min_bars_var = tk.StringVar(value="8")
        ttk.Combobox(
            cue_grid, textvariable=self.cuedetr_min_bars_var,
            values=("4", "8", "16", "32"), state="readonly", width=10
        ).grid(row=2, column=1, sticky="ew", pady=2)
        self.cuedetr_status_label = ttk.Label(
            settings_frame,
            text="正在检测 CUE-DETR…",
            style="Muted.TLabel", wraplength=280,
        )
        self.cuedetr_status_label.pack(anchor=tk.W, fill=tk.X, pady=(2, 4))
        cue_actions = ttk.Frame(settings_frame, style="Card.TFrame")
        cue_actions.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(cue_actions, text="检测模型", command=self._detect_cuedetr).pack(side=tk.LEFT)
        ttk.Button(
            cue_actions, text="分析全部 cue",
            command=lambda: self._analyze_cuedetr(force=False, automatic=False),
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(
            settings_frame,
            text=(
                "旧的 novelty/规则切点已移除。只有 CUE-DETR 可以产生 IN/OUT 候选；"
                "Beat This! 只把预测吸附到 downbeat，All-In-One 只提供段落标签。"
            ),
            style="Muted.TLabel", wraplength=280,
        ).pack(anchor=tk.W, pady=(0, 10))

        ttk.Separator(settings_frame).pack(fill=tk.X, pady=10)
        ttk.Label(settings_frame, text="MuQ 风格与排序", style="Now.TLabel").pack(
            anchor=tk.W, pady=(0, 6)
        )
        self.muq_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            settings_frame,
            text="使用 MuQ-large-msd-iter 参与选歌与 cue 配对评分",
            variable=self.muq_enabled_var,
            command=lambda: self.engine.set_muq_enabled(self.muq_enabled_var.get()),
        ).pack(anchor=tk.W)
        self.auto_muq_sort_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            settings_frame,
            text="MuQ 分析完成后自动重排未播放队列",
            variable=self.auto_muq_sort_var,
        ).pack(anchor=tk.W, pady=(3, 0))
        self.preload_pair_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            settings_frame,
            text="启用滑动窗口预加载",
            variable=self.preload_pair_var,
            command=self._schedule_preload,
        ).pack(anchor=tk.W, pady=(3, 0))

        preload_grid = ttk.Frame(settings_frame, style="Card.TFrame")
        preload_grid.pack(fill=tk.X, pady=(6, 4))
        preload_grid.grid_columnconfigure(1, weight=1)
        ttk.Label(preload_grid, text="窗口轨数", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=2
        )
        self.preload_window_var = tk.StringVar(value="3")
        ttk.Combobox(
            preload_grid,
            textvariable=self.preload_window_var,
            values=("2", "3", "4", "5"),
            state="readonly",
            width=8,
        ).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(preload_grid, text="内存上限", style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=2
        )
        self.preload_memory_var = tk.StringVar(value="1024")
        ttk.Combobox(
            preload_grid,
            textvariable=self.preload_memory_var,
            values=("512", "768", "1024", "1536", "2048"),
            state="readonly",
            width=8,
        ).grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Label(preload_grid, text="截止保护", style="Muted.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=2
        )
        self.preload_deadline_var = tk.StringVar(value="90")
        ttk.Combobox(
            preload_grid,
            textvariable=self.preload_deadline_var,
            values=("30", "45", "60", "90", "120"),
            state="readonly",
            width=8,
        ).grid(row=2, column=1, sticky="ew", pady=2)
        ttk.Label(
            settings_frame,
            text=(
                "MuQ 排序会同时考虑全局风格、Outro→Intro、局部轨迹、BPM、"
                "能量、音色和 All-In-One 段落方向；滑动窗口完整渲染一首热下一轨，"
                "并只对最近未来 pair 做轻量同步与最佳 OUT/IN cue 规划。"
                "内存上限单位为 MB，截止保护单位为秒。"
            ),
            style="Muted.TLabel",
            wraplength=280,
        ).pack(anchor=tk.W, pady=(4, 10))

        ttk.Label(settings_frame, text="Beat This! 模型", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        self.model_var = tk.StringVar(value="final0 · 高精度")
        ttk.Combobox(
            settings_frame,
            textvariable=self.model_var,
            values=(
                "final0 · 高精度",
                "final1 · 高精度备用",
                "small0 · 快速低显存",
            ),
            state="readonly",
        ).pack(fill=tk.X, pady=(4, 4))
        ttk.Label(
            settings_frame,
            text="final0 约 78MB，small0 约 8MB；首次使用会自动下载权重",
            style="Muted.TLabel",
            wraplength=280,
        ).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(settings_frame, text="计算设备", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        self.compute_var = tk.StringVar(value="auto")
        self.compute_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.compute_var,
            values=("auto", "cuda", "cpu", "mps"),
            state="readonly",
        )
        self.compute_combo.pack(fill=tk.X, pady=(4, 4))
        self.compute_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._apply_compute_device(),
        )
        ttk.Label(
            settings_frame,
            text="Beat This!、MuQ、CUE-DETR 可用 auto/cuda；All-In-One 默认 CPU。",
            style="Muted.TLabel",
            wraplength=280,
        ).pack(anchor=tk.W, pady=(0, 12))

        ttk.Label(settings_frame, text="音频输出设备", style="Muted.TLabel").pack(
            anchor=tk.W
        )
        self.output_var = tk.StringVar(value="系统默认")
        self.output_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.output_var,
            values=("系统默认",),
            state="readonly",
        )
        self.output_combo.pack(fill=tk.X, pady=(4, 6))
        ttk.Button(settings_frame, text="刷新设备", command=self._refresh_devices).pack(
            fill=tk.X
        )
        self.settings_saved_label = ttk.Label(
            settings_frame,
            text="配置会自动保存",
            style="Muted.TLabel",
            wraplength=280,
        )
        self.settings_saved_label.pack(anchor=tk.W, fill=tk.X, pady=(8, 0))

        ttk.Separator(settings_frame).pack(fill=tk.X, pady=12)
        ttk.Label(settings_frame, text="匹配详情", style="Now.TLabel").pack(anchor=tk.W)
        self.metrics_label = ttk.Label(
            settings_frame,
            text="等待智能切歌规划",
            style="Muted.TLabel",
            justify=tk.LEFT,
            wraplength=280,
        )
        self.metrics_label.pack(anchor=tk.W, fill=tk.X, pady=(8, 0))

        def update_settings_wrap(event: tk.Event[tk.Misc]) -> None:
            wrap = max(180, event.width - 44)
            for child in settings_frame.winfo_children():
                if isinstance(child, ttk.Label):
                    try:
                        child.configure(wraplength=wrap)
                    except tk.TclError:
                        pass

        settings_frame.bind("<Configure>", update_settings_wrap)

        status_card = ttk.Frame(outer, style="Card.TFrame", padding=10)
        # 先从底部为状态栏预留空间，再让中间区域占用剩余高度；
        # 小屏幕下状态信息也不会被可扩展的 Panedwindow 挤出窗口。
        status_card.pack(
            fill=tk.X, side=tk.BOTTOM, pady=(8, 0), before=middle
        )
        status_card.grid_columnconfigure(0, weight=1)
        self.status_label = ttk.Label(
            status_card,
            text="就绪。添加歌曲后将自动使用 Beat This! 离线分析。",
            style="Muted.TLabel",
            justify=tk.LEFT,
        )
        self.status_label.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        self.cpu_label = ttk.Label(
            status_card,
            text="Audio CPU 0%",
            style="Muted.TLabel",
        )
        self.cpu_label.grid(row=0, column=1, sticky="e")

        self.analysis_progress_text = ttk.Label(
            status_card,
            text="分析进度：空闲",
            style="Muted.TLabel",
            justify=tk.LEFT,
        )
        self.analysis_progress_text.grid(
            row=1, column=0, sticky="ew", pady=(7, 3), padx=(0, 12)
        )
        self.analysis_progress_percent = ttk.Label(
            status_card,
            text="0%",
            style="Metric.TLabel",
        )
        self.analysis_progress_percent.grid(row=1, column=1, sticky="e", pady=(7, 3))
        self.analysis_progress_var = tk.DoubleVar(value=0.0)
        self.analysis_progress_bar = ttk.Progressbar(
            status_card,
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=100.0,
            variable=self.analysis_progress_var,
        )
        self.analysis_progress_bar.grid(row=2, column=0, columnspan=2, sticky="ew")

        status_card.bind(
            "<Configure>",
            lambda event: [
                self.status_label.configure(wraplength=max(220, event.width - 180)),
                self.analysis_progress_text.configure(
                    wraplength=max(220, event.width - 180)
                ),
            ],
        )

    def _checkpoint_name(self) -> str:
        return self.model_var.get().split("·")[0].strip()

    def _bars_value(self) -> int:
        value = self.bars_var.get()
        return 0 if value == "自动" else int(value)

    def _apply_bars(self) -> None:
        self.engine.set_crossfade_bars(self._bars_value())

    def _restore_bars_value(self) -> int:
        value = self.restore_var.get()
        if value == "自动":
            return -1
        if value == "关闭":
            return 0
        return int(value.split()[0])

    def _apply_restore(self) -> None:
        self.engine.set_tempo_restore_bars(self._restore_bars_value())

    def _apply_style(self) -> None:
        self.engine.set_mix_style(self.style_var.get())

    def _apply_human_style(self) -> None:
        self.engine.set_human_style_mode(self.human_style_var.get())

    def _apply_human_candidates(self) -> None:
        self.engine.set_human_candidate_count(int(self.human_candidates_var.get()))

    def _apply_policy(self) -> None:
        self.engine.set_automix_policy(self.policy_var.get())

    def _apply_transition_engine(self) -> None:
        self.engine.set_transition_engine(self.transition_engine_var.get())

    def _apply_stretch_backend(self) -> None:
        self.engine.set_time_stretch_backend(self.stretch_backend_var.get())

    def _choose_rubberband(self) -> None:
        filename = filedialog.askopenfilename(
            title="选择 Rubber Band 命令行程序",
            filetypes=(
                ("Rubber Band CLI", "rubberband.exe rubberband-r3.exe rubberband"),
                ("可执行文件", "*.exe"),
                ("所有文件", "*.*"),
            ),
        )
        if filename:
            self.rb_path_var.set(filename)
            self._apply_rubberband_path()

    def _apply_rubberband_path(self) -> None:
        value = self.rb_path_var.get().strip().strip('"')
        if value:
            os.environ["AUTODJ_RUBBERBAND"] = value
        else:
            os.environ.pop("AUTODJ_RUBBERBAND", None)
        self._detect_rubberband()

    def _clear_rubberband_path(self) -> None:
        self.rb_path_var.set("")
        os.environ.pop("AUTODJ_RUBBERBAND", None)
        self._detect_rubberband()

    def _detect_rubberband(self) -> None:
        explicit = self.rb_path_var.get().strip() or None
        result = rubberband_probe(explicit)
        executable = result.get("executable")
        if result.get("ok") and executable:
            self.rb_status_label.configure(
                text=f"已启用 Rubber Band R3：{executable}",
                foreground="#7ee787",
            )
            self._log(f"Rubber Band R3 已启用：{executable}")
        elif executable:
            self.rb_status_label.configure(
                text=f"找到文件但启动失败：{result.get('message', '未知错误')}",
                foreground="#ffb86c",
            )
        else:
            self.rb_status_label.configure(
                text=(
                    "未检测到。可选择 rubberband.exe，或设置 "
                    "AUTODJ_RUBBERBAND / RUBBERBAND_EXE / RUBBERBAND_PATH。"
                ),
                foreground="#ffb86c",
            )

    def _apply_allin1_model(self) -> None:
        self.engine.set_allin1_model(self.allin1_model_var.get())
        self.allin1_profiles.clear()
        self.engine.set_preloaded_allin1_profiles({})
        self.engine.clear_preload()
        self._refresh_tree()
        self._log("All-In-One 模型已更改，请点击‘分析全部结构’重新生成缓存。")

    def _detect_allin1(self) -> None:
        result = probe_allinone()
        self.allin1_probe_result = dict(result)
        if result.get("ok"):
            self.allin1_status_label.configure(
                text=(
                    f"All-In-One {result.get('allin1_version', '')} 已就绪 · "
                    f"主进程后台线程 · NATTEN {result.get('natten_backend', '')} · "
                    f"{result.get('message', '')}"
                ),
                foreground="#7ee787",
            )
        else:
            self.allin1_status_label.configure(
                text=(
                    "当前主环境未安装 All-In-One。运行 python install_allinone.py；"
                    f"诊断：{result.get('message', '')}"
                ),
                foreground="#ffb86c",
            )

    def _detect_cuedetr(self) -> None:
        result = probe_cuedetr()
        self.cuedetr_probe_result = dict(result)
        if result.get("ok"):
            self.cuedetr_status_label.configure(
                text=(
                    f"CUE-DETR 依赖已就绪 · Transformers {result.get('transformers', '')} · "
                    f"CUDA {'可用' if result.get('cuda') else '不可用'}；首次分析下载权重"
                ),
                foreground="#7ee787",
            )
        else:
            error_type = str(result.get("error_type", ""))
            if error_type == "missing_dependency":
                text = (
                    f"CUE-DETR 缺少依赖：{result.get('message', '')}；"
                    "运行 python install_cuedetr.py"
                )
            else:
                text = (
                    f"CUE-DETR 依赖冲突：{result.get('message', '')}；"
                    "运行 python verify_cuedetr.py 查看详细诊断"
                )
            self.cuedetr_status_label.configure(
                text=text,
                foreground="#ffb86c",
            )

    def _analyze_cuedetr(self, force: bool = False, automatic: bool = False) -> None:
        if self.cuedetr_analysis_active:
            if not automatic:
                messagebox.showinfo("正在分析", "CUE-DETR cue 分析正在运行。")
            return
        if not self.tracks:
            return
        self.cuedetr_analysis_active = True
        tracks_snapshot = list(self.tracks)
        snapshot_paths = tuple(track.path for track in tracks_snapshot)
        model = self.cuedetr_model_var.get().strip() or "disco-eth/cue-detr"
        device = self.cuedetr_device_var.get()
        sensitivity = float(self.cuedetr_sensitivity_var.get())
        min_bars = int(self.cuedetr_min_bars_var.get())
        self._log(
            f"CUE-DETR 正在批量搜索 cue · {model} · {device} · "
            f"灵敏度 {sensitivity:.2f} · 间隔 {min_bars} 小节"
        )
        self.ui_events.put(("analysis_progress", ("CUE-DETR", -1, len(tracks_snapshot), "加载官方模型")))

        def worker() -> None:
            try:
                analyzer = CueDETRAnalyzer(
                    model_name=model, device=device, sensitivity=sensitivity,
                    min_bars=min_bars, batch_size=6,
                )
                profiles = analyzer.analyze_many(
                    tracks_snapshot,
                    status=lambda text: self.ui_events.put(("status", text)),
                    progress=lambda current, total, detail: self.ui_events.put(
                        ("analysis_progress", ("CUE-DETR", current, total, detail))
                    ),
                    force=force,
                )
                self.ui_events.put(("cuedetr_done", (snapshot_paths, profiles, automatic)))
            except Exception as exc:
                self.ui_events.put(("cuedetr_error", (str(exc), automatic)))

        threading.Thread(target=worker, daemon=True, name="CUE-DETR-Analyzer").start()

    def _analyze_allin1(self, force: bool = False, automatic: bool = False) -> None:
        if self.allin1_analysis_active:
            if not automatic:
                messagebox.showinfo("正在分析", "All-In-One 结构分析正在运行。")
            return
        if not self.allin1_enabled_var.get():
            if automatic:
                self._continue_after_allin1()
            else:
                messagebox.showinfo("未启用", "请先勾选启用 All-In-One 功能段模型。")
            return
        if not self.tracks:
            if not automatic:
                messagebox.showinfo("没有歌曲", "请先添加歌曲。")
            return

        self.allin1_analysis_active = True
        tracks_snapshot = list(self.tracks)
        snapshot_paths = tuple(track.path for track in tracks_snapshot)
        model = self.allin1_model_var.get()
        device = self.allin1_device_var.get()
        self._log(
            f"All-In-One 正在主进程后台线程批量分析 · {model} · {device}；"
            "分析期间暂停 MuQ 排序和预加载…"
        )
        self.ui_events.put(
            ("analysis_progress", ("All-In-One", -1, len(tracks_snapshot), "加载 Demucs 与结构模型"))
        )

        def worker() -> None:
            try:
                analyzer = AllInOneAnalyzer(model=model, device=device)
                profiles = analyzer.analyze_many(
                    [track.path for track in tracks_snapshot],
                    status=lambda text: self.ui_events.put(("status", text)),
                    progress=lambda current, total, detail: self.ui_events.put(
                        ("analysis_progress", ("All-In-One", current, total, detail))
                    ),
                    force=force,
                )
                self.ui_events.put(("allin1_done", (snapshot_paths, profiles, automatic)))
            except Exception as exc:
                self.ui_events.put(("allin1_error", (str(exc), automatic)))

        threading.Thread(
            target=worker, daemon=True, name="All-In-One-InProcess"
        ).start()

    def _continue_after_allin1(self) -> None:
        if self.cuedetr_enabled_var.get():
            self._log("结构标签完成，正在运行 CUE-DETR 专业 cue 检测…")
            self._analyze_cuedetr(force=False, automatic=True)
        else:
            self._continue_after_cuedetr()

    def _continue_after_cuedetr(self) -> None:
        if self.auto_muq_sort_var.get() and self.muq_enabled_var.get():
            self._log("歌曲结构分析完成，正在进行 MuQ 风格分析与排序…")
            self._rank_with_muq(automatic=True)
        else:
            self._log("Beat This!、All-In-One 与 CUE-DETR 分析完成。")
            self._schedule_preload()

    def _set_effect_strength(self, value: str) -> None:
        number = float(value)
        self.effect_label.configure(text=f"{number:.0f}%")
        self.engine.set_effect_strength(number / 100.0)

    def _apply_engine_settings(self) -> None:
        """把 GUI 当前设置同步到引擎，播放和预加载共用。"""
        self.engine.set_auto_mix(self.auto_mix_var.get())
        self.engine.set_volume(self.volume_var.get() / 100.0)
        self.engine.set_crossfade_bars(self._bars_value())
        self.engine.set_max_stretch_percent(self.stretch_var.get())
        self.engine.set_tempo_restore_bars(self._restore_bars_value())
        self.engine.set_mix_style(self.style_var.get())
        self.engine.set_effect_strength(self.effect_var.get() / 100.0)
        self.engine.set_human_style_mode(self.human_style_var.get())
        self.engine.set_human_candidate_count(int(self.human_candidates_var.get()))
        self.engine.set_automix_policy(self.policy_var.get())
        self.engine.set_transition_engine(self.transition_engine_var.get())
        self.engine.set_time_stretch_backend(self.stretch_backend_var.get())
        self.engine.set_muq_enabled(self.muq_enabled_var.get())
        self.engine.set_muq_device(self.compute_var.get())
        self.engine.set_preloaded_muq_profiles(self.muq_profiles)
        self.engine.set_allin1_enabled(self.allin1_enabled_var.get())
        self.engine.set_allin1_device(self.allin1_device_var.get())
        self.engine.set_allin1_model(self.allin1_model_var.get())
        self.engine.set_preloaded_allin1_profiles(self.allin1_profiles)
        self.cuedetr_enabled_var.set(True)
        self.engine.set_cuedetr_enabled(True)
        self.engine.set_cuedetr_device(self.cuedetr_device_var.get())
        self.engine.set_cuedetr_model(self.cuedetr_model_var.get())
        self.engine.set_cuedetr_sensitivity(float(self.cuedetr_sensitivity_var.get()))
        self.engine.set_cuedetr_min_bars(int(self.cuedetr_min_bars_var.get()))
        self.engine.set_preloaded_cuedetr_profiles(self.cuedetr_profiles)
        self.engine.set_preload_window_tracks(int(self.preload_window_var.get()))
        self.engine.set_preload_memory_mb(int(self.preload_memory_var.get()))
        self.engine.set_preload_deadline_seconds(
            float(self.preload_deadline_var.get())
        )

    def _schedule_preload(self, delay_ms: int = 450) -> None:
        if self._preload_after_id is not None:
            try:
                self.after_cancel(self._preload_after_id)
            except tk.TclError:
                pass
            self._preload_after_id = None
        if (
            not self.preload_pair_var.get()
            or not self.tracks
            or self.analysis_active
            or self.allin1_analysis_active
            or self.muq_ranking_active
            or self.cuedetr_analysis_active
            or self.engine.get_status().get("playing")
        ):
            return
        self._preload_after_id = self.after(delay_ms, self._preload_selected_pair)

    def _preload_selected_pair(self) -> None:
        self._preload_after_id = None
        if (
            not self.preload_pair_var.get()
            or not self.tracks
            or self.analysis_active
            or self.allin1_analysis_active
            or self.muq_ranking_active
            or self.cuedetr_analysis_active
            or self.engine.get_status().get("playing")
        ):
            return
        try:
            self._apply_engine_settings()
            index = self._selected_index()
            started = self.engine.preload_pair(list(self.tracks), start_index=index)
            if started:
                next_title = (
                    self.tracks[index + 1].title
                    if index + 1 < len(self.tracks)
                    else "队列末尾"
                )
                self._log(
                    f"后台预加载：{self.tracks[index].title} / 下一首 {next_title}"
                )
        except Exception as exc:
            self._log(f"启动预加载失败：{exc}")

    def _seek_to(self, seconds: float) -> None:
        status = self.engine.get_status()
        if not status.get("current"):
            return
        try:
            actual = self.engine.seek(seconds, snap_to_beat=True)
            self._log(f"已跳转到 {format_time(actual)}，沿用已缓存结果，不重新计算")
        except Exception as exc:
            self._log(f"跳转失败：{exc}")

    def _apply_compute_device(self) -> None:
        device = self.compute_var.get()
        try:
            self.engine.set_muq_device(device)
            self.engine.clear_preload()
            self._log(f"Beat This! / MuQ 计算设备已切换为 {device}；CUE-DETR 与 All-In-One 设备独立设置")
        except Exception as exc:
            self._log(f"切换计算设备失败：{exc}")

    def _refresh_devices(self) -> None:
        try:
            entries = {"系统默认": None}
            for index, name in self.engine.output_devices():
                entries[f"{index}: {name}"] = index
            self.device_map = entries
            self.output_combo["values"] = tuple(entries.keys())
            if self.output_var.get() not in entries:
                self.output_var.set("系统默认")
        except Exception as exc:
            self._log(f"无法枚举音频设备：{exc}")

    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(title="选择歌曲", filetypes=AUDIO_FILETYPES)
        if paths:
            self._analyze_paths([Path(path) for path in paths], force=False)

    def _analyze_paths(self, paths: list[Path], force: bool) -> None:
        if self.analysis_active or self.allin1_analysis_active or self.cuedetr_analysis_active or self.muq_ranking_active:
            messagebox.showinfo("正在分析", "请等待当前模型分析完成。")
            return
        self.analysis_active = True
        total = len(paths)
        self.ui_events.put(("analysis_progress", ("Beat This!", -1, total, "准备模型")))
        LOGGER.info("Beat This! 批量分析开始，共 %d 首", total)

        def worker() -> None:
            try:
                analyzer = BeatThisAnalyzer(
                    checkpoint=self._checkpoint_name(),
                    device=self.compute_var.get(),
                )
                for index, path in enumerate(paths):
                    self.ui_events.put(
                        ("analysis_progress", ("Beat This!", index, total, path.name))
                    )
                    try:
                        result = analyzer.analyze(
                            path,
                            status=lambda text, i=index, name=path.name: [
                                self.ui_events.put(("status", text)),
                                self.ui_events.put(
                                    ("analysis_progress", ("Beat This!", i + 0.35, total, name))
                                ),
                            ],
                            force=force,
                        )
                        self.ui_events.put(("track", result))
                    except Exception as exc:
                        self.ui_events.put(("error", f"{path.name}：{exc}"))
                    finally:
                        self.ui_events.put(
                            ("analysis_progress", ("Beat This!", index + 1, total, path.name))
                        )
            finally:
                LOGGER.info("Beat This! 批量分析阶段结束")
                self.ui_events.put(("analysis_done", None))

        threading.Thread(target=worker, daemon=True, name="BeatThis-Analysis").start()

    def _reanalyze_selected(self) -> None:
        selection = self.tree.selection()
        if selection:
            self._analyze_paths([Path(self.tracks[int(selection[0])].path)], force=True)

    def _insert_or_replace_track(self, track: TrackAnalysis) -> None:
        for index, existing in enumerate(self.tracks):
            if Path(existing.path) == Path(track.path):
                self.tracks[index] = track
                self._refresh_tree()
                self._schedule_preload()
                return
        self.tracks.append(track)
        self._refresh_tree()
        if len(self.tracks) == 1:
            self.tree.selection_set("0")
        self._schedule_preload()

    def _refresh_tree(self, status: dict[str, object] | None = None) -> None:
        status = status or self.engine.get_status()
        selected = self.tree.selection()
        selected_path = None
        if selected:
            old_index = int(selected[0])
            if 0 <= old_index < len(self.tracks):
                selected_path = self.tracks[old_index].path
        for item in self.tree.get_children():
            self.tree.delete(item)

        current_index = int(status.get("index", -1))
        playing = bool(status.get("playing", False))
        transition_start = status.get("transition_start")
        next_entry = status.get("next_entry")
        preload_current = str(status.get("preload_current_path") or "")
        preload_next = str(status.get("preload_next_path") or "")
        preload_loading = bool(status.get("preload_loading"))
        preload_ready = bool(status.get("preload_ready"))
        warm_ready = set(status.get("warm_ready_paths") or ())
        warm_loading = set(status.get("warm_loading_paths") or ())
        for index, track in enumerate(self.tracks):
            state = ""
            mix = ""
            if playing and index == current_index:
                state = "播放中"
                if transition_start is not None:
                    mix = f"OUT {format_time(float(transition_start))}"
            elif playing and index == current_index + 1:
                state = "下一首"
                if next_entry is not None:
                    original_bpm = float(status.get("next_original_bpm", track.bpm))
                    mix = f"IN {format_time(float(next_entry))} → {original_bpm:.0f}"
            elif not playing and track.path == preload_current:
                state = "预加载中" if preload_loading else ("已预载" if preload_ready else "")
            elif not playing and track.path == preload_next:
                state = "下一首预载" if preload_loading or preload_ready else ""
            elif track.path in warm_loading:
                state = "窗口预热"
            elif track.path in warm_ready:
                state = "暖轨就绪"
            structure_profile = self.allin1_profiles.get(track.path)
            cue_profile = self.cuedetr_profiles.get(track.path)
            cue_suffix = (
                f" · CUE {len(cue_profile.cue_times)}"
                if cue_profile and cue_profile.available else ""
            )
            if structure_profile and structure_profile.available:
                structure_text = "/".join(
                    label.upper() for label in structure_profile.unique_labels[:4]
                ) + cue_suffix
            elif self.allin1_analysis_active or self.cuedetr_analysis_active:
                structure_text = "分析中…"
            else:
                structure_text = (cue_suffix.strip(" ·") or "—")
            self.tree.insert(
                "",
                tk.END,
                iid=str(index),
                values=(
                    index + 1,
                    track.title,
                    f"{track.bpm:.1f}",
                    f"{track.beats_per_bar}/4",
                    format_time(track.duration),
                    structure_text,
                    (f"组 {self.muq_groups.get(track.path)}" if track.path in self.muq_groups else "—"),
                    (
                        f"{self.muq_pair_scores[track.path].total * 100:.0f}分"
                        if track.path in self.muq_pair_scores else "—"
                    ),
                    mix,
                    state,
                ),
            )
            if track.path == selected_path:
                self.tree.selection_set(str(index))

    def _rank_with_muq(self, automatic: bool = False) -> None:
        if self.muq_ranking_active:
            return
        if self.analysis_active or self.allin1_analysis_active or self.cuedetr_analysis_active:
            if not automatic:
                messagebox.showinfo("正在分析", "请等待 Beat This! / All-In-One / CUE-DETR 分析完成。")
            return
        if len(self.tracks) < 2:
            if not automatic:
                messagebox.showinfo("歌曲不足", "至少添加两首歌曲后才能进行 MuQ 排序。")
            self._schedule_preload()
            return

        self.muq_ranking_active = True
        tracks_snapshot = list(self.tracks)
        snapshot_paths = tuple(track.path for track in tracks_snapshot)
        structure_snapshot = dict(self.allin1_profiles)
        status = self.engine.get_status()
        playing = bool(status.get("playing"))
        current_path = str(status.get("current_path") or "")
        next_path = str(status.get("next_path") or "")
        transitioning = bool(status.get("transitioning"))
        selected_index = self._selected_index()
        self._log("MuQ 正在分析风格并优化播放顺序…")
        self.ui_events.put(
            ("analysis_progress", ("MuQ", 0, len(tracks_snapshot), "加载模型"))
        )

        def worker() -> None:
            try:
                analyzer = MuQAnalyzer(device=self.compute_var.get())
                profile_map: dict[str, MuQProfile] = {}
                profiles_in_snapshot: list[MuQProfile] = []
                total_tracks = len(tracks_snapshot)
                for track_index, track in enumerate(tracks_snapshot):
                    self.ui_events.put(
                        ("analysis_progress", ("MuQ", track_index, total_tracks, track.title))
                    )
                    profile = analyzer.analyze(
                        track.path,
                        status=lambda text: self.ui_events.put(("status", text)),
                        progress=lambda fraction, detail, i=track_index, total=total_tracks: self.ui_events.put(
                            ("analysis_progress", ("MuQ", i + fraction, total, detail))
                        ),
                    )
                    profile_map[track.path] = profile
                    profiles_in_snapshot.append(profile)
                    self.ui_events.put(
                        ("analysis_progress", ("MuQ", track_index + 1, total_tracks, track.title))
                    )

                group_values = style_clusters(profiles_in_snapshot)
                group_map = {
                    track.path: int(group)
                    for track, group in zip(tracks_snapshot, group_values)
                }

                fixed_prefix: list[TrackAnalysis] = []
                segment = list(tracks_snapshot)
                segment_start = selected_index
                if playing and current_path:
                    path_to_index = {track.path: i for i, track in enumerate(tracks_snapshot)}
                    current_index = path_to_index.get(current_path, 0)
                    anchor_index = current_index
                    # 过渡已经开始时，当前 B 不能再替换；从 B 之后排序。
                    if transitioning and next_path in path_to_index:
                        anchor_index = path_to_index[next_path]
                    fixed_prefix = tracks_snapshot[:anchor_index]
                    segment = tracks_snapshot[anchor_index:]
                    segment_start = 0

                segment_profiles = [profile_map[track.path] for track in segment]
                segment_structures = [
                    structure_snapshot.get(track.path, AllInOneProfile())
                    for track in segment
                ]
                order, _scores = rank_playlist(
                    segment,
                    segment_profiles,
                    start_index=segment_start,
                    structures=segment_structures,
                )
                ordered_segment = [segment[index] for index in order]
                ordered_tracks = fixed_prefix + ordered_segment

                pair_map: dict[str, PairScore] = {}
                edge_scores: list[float] = []
                for outgoing, incoming in zip(ordered_tracks, ordered_tracks[1:]):
                    pair = transition_compatibility(
                        outgoing,
                        incoming,
                        profile_map[outgoing.path],
                        profile_map[incoming.path],
                        structure_snapshot.get(outgoing.path),
                        structure_snapshot.get(incoming.path),
                    )
                    pair_map[outgoing.path] = pair
                    edge_scores.append(pair.total)
                average = sum(edge_scores) / max(len(edge_scores), 1)
                worst = min(edge_scores, default=1.0)
                self.ui_events.put(
                    (
                        "muq_ranked",
                        (
                            snapshot_paths,
                            ordered_tracks,
                            profile_map,
                            group_map,
                            pair_map,
                            average,
                            worst,
                        ),
                    )
                )
            except Exception as exc:
                self.ui_events.put(("muq_error", str(exc)))

        threading.Thread(target=worker, daemon=True, name="MuQ-Ranking").start()

    def _remove_selected(self) -> None:
        if self.engine.get_status()["playing"]:
            messagebox.showinfo("播放中", "停止播放后再修改当前队列。")
            return
        selection = self.tree.selection()
        if selection:
            removed = self.tracks[int(selection[0])]
            del self.tracks[int(selection[0])]
            self.muq_profiles.pop(removed.path, None)
            self.muq_groups.pop(removed.path, None)
            self.muq_pair_scores.pop(removed.path, None)
            self.allin1_profiles.pop(removed.path, None)
            self.cuedetr_profiles.pop(removed.path, None)
            self.engine.clear_preload()
            self._refresh_tree()
            self._schedule_preload()

    def _clear(self) -> None:
        self._stop()
        self.tracks.clear()
        self.muq_profiles.clear()
        self.muq_groups.clear()
        self.muq_pair_scores.clear()
        self.allin1_profiles.clear()
        self.cuedetr_profiles.clear()
        self.engine.clear_preload()
        self._refresh_tree()

    def _selected_index(self) -> int:
        selection = self.tree.selection()
        return int(selection[0]) if selection else 0

    def _play_or_resume(self) -> None:
        status = self.engine.get_status()
        if status["playing"] and status["paused"]:
            self.engine.resume()
            self._log("继续播放。")
            return
        if status["playing"] or self.starting_playback:
            return
        if not self.tracks:
            messagebox.showinfo("没有歌曲", "请先添加并分析至少一首歌曲。")
            return

        index = self._selected_index()
        playlist = list(self.tracks)
        missing_cues = [
            track.title
            for track in playlist[index:]
            if not self.cuedetr_profiles.get(track.path, CueDETRProfile()).available
        ]
        if missing_cues:
            self._log("播放已阻止：必须先完成 CUE-DETR cue 分析。")
            messagebox.showinfo(
                "需要 CUE-DETR",
                "以下歌曲尚未完成 CUE-DETR cue 分析：\n"
                + "\n".join(missing_cues[:8])
                + ("\n…" if len(missing_cues) > 8 else ""),
            )
            self._analyze_cuedetr(force=False, automatic=False)
            return
        device = self.device_map.get(self.output_var.get())
        self.starting_playback = True
        self.play_button.state(["disabled"])
        self._log(f"正在提取智能匹配特征：{playlist[index].title}")

        self._apply_engine_settings()

        def worker() -> None:
            try:
                self.engine.start_playlist(playlist, start_index=index, device=device)
                self.ui_events.put(("play_started", None))
            except Exception as exc:
                self.ui_events.put(("play_error", str(exc)))

        threading.Thread(target=worker, daemon=True, name="AutoDJ-Start").start()

    def _pause(self) -> None:
        self.engine.pause()
        self._log("已暂停。")

    def _stop(self) -> None:
        try:
            self.engine.stop()
        except Exception as exc:
            self._log(f"停止播放时出错：{exc}")
        self.last_engine_index = -1
        self.last_plan_signature = object()
        self.last_preload_signature = object()
        self._refresh_tree()
        self._log("已停止。")
        self._schedule_preload()

    def _next(self) -> None:
        self.engine.request_next()

    def _set_stretch(self, value: str) -> None:
        number = float(value)
        self.stretch_label.configure(text=f"±{number:.1f}%")
        self.engine.set_max_stretch_percent(number)

    def _log(self, text: str) -> None:
        message = str(text)
        self.status_label.configure(text=message)
        LOGGER.info(message)

    def _set_analysis_progress(
        self,
        phase: str,
        current: float,
        total: float,
        detail: str = "",
    ) -> None:
        total_value = max(float(total), 1.0)
        text = f"分析进度 · {phase}"
        if detail:
            text += f" · {detail}"

        if float(current) < 0.0:
            self.analysis_progress_bar.configure(mode="indeterminate")
            self.analysis_progress_bar.start(12)
            self.analysis_progress_percent.configure(text="…")
            self.analysis_progress_text.configure(text=text)
            return

        self.analysis_progress_bar.stop()
        self.analysis_progress_bar.configure(mode="determinate", maximum=100.0)
        current_value = min(max(float(current), 0.0), total_value)
        percent = 100.0 * current_value / total_value
        self.analysis_progress_var.set(percent)
        self.analysis_progress_percent.configure(text=f"{percent:.0f}%")
        count_text = (
            f"{current_value:.1f}/{total_value:.0f}"
            if current_value % 1
            else f"{int(current_value)}/{int(total_value)}"
        )
        self.analysis_progress_text.configure(text=f"{text} · {count_text}")

    def _finish_analysis_progress(self, text: str = "全部分析完成") -> None:
        self.analysis_progress_bar.stop()
        self.analysis_progress_bar.configure(mode="determinate", maximum=100.0)
        self.analysis_progress_var.set(100.0)
        self.analysis_progress_percent.configure(text="100%")
        self.analysis_progress_text.configure(text=f"分析进度 · {text}")

    def _poll_ui_events(self) -> None:
        while True:
            try:
                kind, payload = self.ui_events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self._log(str(payload))
            elif kind == "analysis_progress":
                phase, current, total, detail = payload  # type: ignore[misc]
                self._set_analysis_progress(
                    str(phase), float(current), float(total), str(detail)
                )
            elif kind == "track":
                self._insert_or_replace_track(payload)  # type: ignore[arg-type]
            elif kind == "error":
                self._log(str(payload))
                messagebox.showerror("Beat This! 分析失败", str(payload))
            elif kind == "analysis_done":
                self.analysis_active = False
                if self.allin1_enabled_var.get():
                    self._log("Beat This! 完成，正在运行 All-In-One 功能段分析…")
                    self._analyze_allin1(force=False, automatic=True)
                else:
                    self._continue_after_allin1()
            elif kind == "allin1_done":
                snapshot_paths, profiles, automatic = payload  # type: ignore[misc]
                self.allin1_analysis_active = False
                current_paths = tuple(track.path for track in self.tracks)
                if set(snapshot_paths) != set(current_paths):
                    self._log("队列在结构分析期间发生变化，重新分析 All-In-One…")
                    self._analyze_allin1(force=False, automatic=True)
                    continue
                self.allin1_profiles.update(dict(profiles))
                self.engine.set_preloaded_allin1_profiles(self.allin1_profiles)
                self._refresh_tree()
                available = sum(
                    1 for profile in self.allin1_profiles.values() if profile.available
                )
                self.allin1_status_label.configure(
                    text=f"All-In-One 已完成 {available}/{len(self.tracks)} 首结构分析",
                    foreground="#7ee787",
                )
                if automatic:
                    self._continue_after_allin1()
                else:
                    self._log(f"All-In-One 结构分析完成：{available} 首")
                    self._finish_analysis_progress("All-In-One 分析完成")
                    self._schedule_preload(delay_ms=150)
            elif kind == "allin1_error":
                message, automatic = payload  # type: ignore[misc]
                self.allin1_analysis_active = False
                self.allin1_status_label.configure(
                    text=f"All-In-One 失败：{message}", foreground="#ffb86c"
                )
                self._log(f"All-In-One 失败，保留本地结构估计：{message}")
                if not automatic:
                    messagebox.showerror("All-In-One 分析失败", str(message))
                if automatic:
                    self._continue_after_allin1()
                else:
                    self._schedule_preload()
            elif kind == "cuedetr_done":
                snapshot_paths, profiles, automatic = payload  # type: ignore[misc]
                self.cuedetr_analysis_active = False
                current_paths = tuple(track.path for track in self.tracks)
                if set(snapshot_paths) != set(current_paths):
                    self._log("队列在 CUE-DETR 分析期间变化，正在重新分析…")
                    self._analyze_cuedetr(force=False, automatic=True)
                    continue
                self.cuedetr_profiles.update(dict(profiles))
                self.engine.set_preloaded_cuedetr_profiles(self.cuedetr_profiles)
                count = sum(len(profile.cue_times) for profile in self.cuedetr_profiles.values())
                self.cuedetr_status_label.configure(
                    text=f"CUE-DETR 已完成 {len(self.cuedetr_profiles)} 首 · 共 {count} 个 downbeat cue",
                    foreground="#7ee787",
                )
                self.engine.clear_preload()
                if automatic:
                    self._continue_after_cuedetr()
                else:
                    self._log(f"CUE-DETR cue 分析完成：共 {count} 个")
                    self._finish_analysis_progress("CUE-DETR 分析完成")
                    self._schedule_preload(delay_ms=120)
            elif kind == "cuedetr_error":
                message, automatic = payload  # type: ignore[misc]
                self.cuedetr_analysis_active = False
                self.cuedetr_status_label.configure(
                    text=f"CUE-DETR 失败：{message}", foreground="#ffb86c"
                )
                self._log(f"CUE-DETR 失败：{message}")
                if not automatic:
                    messagebox.showerror("CUE-DETR 分析失败", str(message))
                if automatic:
                    self._continue_after_cuedetr()
                else:
                    self._schedule_preload()
            elif kind == "muq_ranked":
                (
                    snapshot_paths,
                    ordered_tracks,
                    profile_map,
                    group_map,
                    pair_map,
                    average,
                    worst,
                ) = payload  # type: ignore[misc]
                self.muq_ranking_active = False
                current_paths = tuple(track.path for track in self.tracks)
                if set(snapshot_paths) != set(current_paths):
                    self._log("队列在 MuQ 分析期间发生变化，正在重新排序…")
                    self._rank_with_muq(automatic=True)
                    continue
                selected_path = None
                selection = self.tree.selection()
                if selection:
                    old_index = int(selection[0])
                    if 0 <= old_index < len(self.tracks):
                        selected_path = self.tracks[old_index].path
                self.muq_profiles = dict(profile_map)
                self.muq_groups = dict(group_map)
                self.muq_pair_scores = dict(pair_map)
                self.tracks = list(ordered_tracks)
                self.engine.set_preloaded_muq_profiles(self.muq_profiles)
                self.engine.set_preloaded_allin1_profiles(self.allin1_profiles)
                self.engine.set_preloaded_cuedetr_profiles(self.cuedetr_profiles)
                if self.engine.get_status().get("playing"):
                    try:
                        self.engine.update_playlist_order(self.tracks)
                    except Exception as exc:
                        self._log(f"播放顺序同步失败：{exc}")
                self._refresh_tree()
                if selected_path:
                    for index, track in enumerate(self.tracks):
                        if track.path == selected_path:
                            self.tree.selection_set(str(index))
                            break
                elif self.tracks:
                    self.tree.selection_set("0")
                self._log(
                    "MuQ 平滑排序完成："
                    f"平均 {average * 100:.0f}分 · 最弱相邻边 {worst * 100:.0f}分"
                )
                self._finish_analysis_progress("分析、排序与队列更新完成")
                self._schedule_preload(delay_ms=150)
            elif kind == "muq_error":
                self.muq_ranking_active = False
                self.analysis_progress_text.configure(text=f"分析进度 · MuQ 失败 · {payload}")
                self._log(f"MuQ 排序失败：{payload}")
                messagebox.showerror("MuQ 排序失败", str(payload))
                self._schedule_preload()
            elif kind == "play_started":
                self.starting_playback = False
                self.play_button.state(["!disabled"])
            elif kind == "play_error":
                self.starting_playback = False
                self.play_button.state(["!disabled"])
                self._log(f"播放启动失败：{payload}")
                messagebox.showerror("播放启动失败", str(payload))

    def _update_match_details(self, status: dict[str, object]) -> None:
        start = status.get("transition_start")
        entry = status.get("next_entry")
        bars = int(status.get("transition_bars", 0))
        score = float(status.get("match_score", 0.0))
        if start is None or entry is None:
            self.plan_label.configure(text="切歌规划：正在分析下一首匹配片段…")
            self.metrics_label.configure(text="等待智能切歌规划")
            self.score_label.configure(text="MATCH\n—")
            return

        style = str(status.get("mix_style", "Club"))
        sync_bpm = float(status.get("next_sync_bpm", 0.0))
        original_bpm = float(status.get("next_original_bpm", 0.0))
        restore_bars = int(status.get("tempo_restore_bars", 0))
        tempo_text = ""
        if sync_bpm > 0 and original_bpm > 0 and abs(sync_bpm - original_bpm) > 0.05:
            tempo_text = f" · BPM {sync_bpm:.1f}→{original_bpm:.1f}/{restore_bars}小节"
        transition_mode = str(status.get("transition_mode", "EQ/Fader"))
        policy_mode = str(status.get("policy_mode", "AutoMix-like"))
        archetype = str(status.get("human_archetype", ""))
        archetype_text = f" · {archetype}" if archetype else ""
        dj_intent = str(status.get("dj_intent", "") or "")
        intent_text = f" · {dj_intent}" if dj_intent else ""
        current_function = str(status.get("current_function_label", "") or "")
        next_function = str(status.get("next_function_label", "") or "")
        function_text = (
            f" · {current_function}→{next_function}"
            if current_function or next_function
            else ""
        )
        bars_text = f"{bars} 小节" if bars > 0 else "无缝裁切"
        self.plan_label.configure(
            text=(
                f"真人化切歌：A OUT {format_time(float(start))}  →  "
                f"B IN {format_time(float(entry))} · {bars_text}{intent_text}{archetype_text}"
                f"{function_text} · {transition_mode} / {policy_mode}{tempo_text}"
            )
        )
        self.score_label.configure(text=f"MATCH\n{score * 100:.0f}")
        metrics = status.get("match_metrics") or {}
        if isinstance(metrics, dict):
            text = (
                f"连续性    {float(metrics.get('continuity', 0.0)) * 100:5.0f}\n"
                f"和声相似  {float(metrics.get('harmonic', 0.0)) * 100:5.0f}\n"
                f"Camelot   {float(metrics.get('key_score', 0.0)) * 100:5.0f}  "
                f"{str(status.get('current_key', '—'))}→{str(status.get('next_key', '—'))}\n"
                f"Cue 对齐  {float(metrics.get('cue_alignment', 0.0)) * 100:5.0f}\n"
                f"Phrase    {float(metrics.get('phrase_alignment', 0.0)) * 100:5.0f}\n"
                f"AIO 边界   {float(metrics.get('allin1_boundary', 0.0)) * 100:5.0f}\n"
                f"AIO 角色   {float(metrics.get('allin1_role_compatibility', 0.0)) * 100:5.0f}  "
                f"{str(status.get('current_function_label', '—'))}→{str(status.get('next_function_label', '—'))}\n"
                f"DJ 意图   {str(status.get('dj_intent', '—'))}\n"
                f"角色路径  {str(status.get('current_role', '—'))}→{str(status.get('next_role', '—'))}→{str(status.get('next_landing_role', '—'))}\n"
                f"结构策略  {float(metrics.get('dj_phrase_policy', 0.0)) * 100:5.0f}\n"
                f"Drop落点  {float(metrics.get('dj_drop_landing', 0.0)) * 100:5.0f}\n"
                f"Drop后出  {float(metrics.get('dj_post_drop', 0.0)) * 100:5.0f}\n"
                f"能量轨迹  {float(metrics.get('dj_energy_arc', 0.0)) * 100:5.0f}\n"
                f"节奏      {float(metrics.get('rhythm', 0.0)) * 100:5.0f}\n"
                f"低频净度  {float(metrics.get('bass_clean', 0.0)) * 100:5.0f}\n"
                f"人声避让  {float(metrics.get('vocal_clean', 0.0)) * 100:5.0f}\n"
                f"EDM 置信  {float(metrics.get('edm_confidence', 0.0)) * 100:5.0f}\n"
                f"MuQ 风格  {float(metrics.get('muq_style', 0.0)) * 100:5.0f}\n"
                f"MuQ 片段  {float(metrics.get('muq_segment', 0.0)) * 100:5.0f}\n"
                f"MuQ 轨迹  {float(metrics.get('muq_trajectory', 0.0)) * 100:5.0f}\n"
                f"首拍微调  {float(metrics.get('micro_align_ms', 0.0)):5.1f} ms\n"
                f"相位锁定  {float(metrics.get('beat_lock_pre_error_ms', 0.0)):4.1f}→{float(metrics.get('beat_lock_post_error_ms', 0.0)):4.1f} ms  "
                f"{float(metrics.get('beat_grid_aligned_beats', 0.0)):2.0f} 拍\n"
                f"小节 Nudge {float(metrics.get('beat_lock_bar_nudges', 0.0)):4.0f} 次  "
                f"置信 {float(metrics.get('beat_lock_confidence', 0.0)) * 100:3.0f}%\n"
                f"真人手法  {str(status.get('human_archetype', '—'))}\n"
                f"真人评分  {float(status.get('human_quality_score', 0.0)) * 100:5.0f}\n"
                f"响度控制  {float(metrics.get('quality_loudness', 0.0)) * 100:5.0f}\n"
                f"频段避让  {float(metrics.get('quality_collision', 0.0)) * 100:5.0f}\n"
                f"频谱连续  {float(metrics.get('quality_continuity', 0.0)) * 100:5.0f}\n"
                f"曲线平滑  {float(metrics.get('quality_smoothness', 0.0)) * 100:5.0f}\n"
                f"立体声稳  {float(metrics.get('quality_stereo', 0.0)) * 100:5.0f}\n"
                f"节拍一致  {float(metrics.get('quality_beat', 0.0)) * 100:5.0f}\n"
                f"双鼓交接  {float(metrics.get('quality_drum_handover', 0.0)) * 100:5.0f}\n"
                f"结构源    {str(status.get('current_structure_source', '—'))}\n"
                f"渲染      {str(status.get('transition_mode', '—'))}\n"
                f"拉伸      {str(status.get('stretch_backend', '—'))}\n"
                f"风格      {str(status.get('mix_style', 'Club'))} · "
                f"{float(status.get('effect_strength', 0.0)) * 100:.0f}%"
            )
            self.metrics_label.configure(text=text)

    def _poll(self) -> None:
        self._poll_ui_events()
        try:
            self.engine.service()
        except Exception as exc:
            self._log(f"后台准备失败：{exc}")
        for message in self.engine.drain_events():
            self._log(message)

        status = self.engine.get_status()
        self.now_label.configure(text=status["current"] or "尚未播放")
        if status.get("next_ready"):
            next_text = f"{status['next']} · 热轨已就绪"
        elif status.get("preload_urgent"):
            next_text = "截止保护：正在最高优先级准备下一首…"
        elif status["next_loading"]:
            next_text = "滑动窗口正在分析与预加载…"
        elif status.get("preload_loading") and not status.get("playing"):
            next_text = "播放前预加载中…"
        else:
            next_text = status["next"] or "—"
        self.next_label.configure(text=f"下一首：{next_text}")
        bpm_now = float(status["bpm"])
        original_bpm = float(status.get("original_bpm", bpm_now))
        restoring = float(status.get("tempo_restore_progress", 0.0))
        if 0.0 < restoring < 1.0 and abs(bpm_now - original_bpm) > 0.05:
            self.bpm_label.configure(text=f"{bpm_now:.1f}\n→{original_bpm:.1f}")
        else:
            self.bpm_label.configure(text=f"{bpm_now:.1f}\nBPM")
        self.time_left.configure(text=format_time(status["position"]))
        self.time_right.configure(text=format_time(status["duration"]))
        self.timeline.render(status)
        self._update_match_details(status)

        beat = int(status["beat_number"])
        self.beat_label.configure(text=f"BEAT\n{beat or '—'}")
        if beat and beat != self.last_beat_number:
            self.beat_label.configure(background="#ff4d6d")
            self.after(100, lambda: self.beat_label.configure(background="#343a40"))
            self.last_beat_number = beat

        warm_count = len(status.get("warm_ready_paths") or ())
        warm_mb = float(status.get("warm_cache_mb", 0.0))
        urgent_text = " · 截止保护" if status.get("preload_urgent") else ""
        self.cpu_label.configure(
            text=(
                f"Audio CPU {status['cpu_load'] * 100:.0f}% · "
                f"暖轨 {warm_count} · {warm_mb:.0f}MB{urgent_text}"
            )
        )
        if status["callback_status"]:
            self.cpu_label.configure(text=f"音频警告：{status['callback_status']}")

        preload_signature = (
            status.get("preload_loading"),
            status.get("preload_ready"),
            status.get("preload_current_path"),
            status.get("preload_next_path"),
            status.get("next_ready"),
            tuple(status.get("warm_ready_paths") or ()),
            tuple(status.get("warm_loading_paths") or ()),
            status.get("preload_urgent"),
        )
        if (
            status["index"] != self.last_engine_index
            or status["plan_signature"] != self.last_plan_signature
            or preload_signature != self.last_preload_signature
        ):
            self.last_engine_index = status["index"]
            self.last_plan_signature = status["plan_signature"]
            self.last_preload_signature = preload_signature
            self._refresh_tree(status)

        self.play_button.configure(text="▶ 继续" if status["paused"] else "▶ 播放")
        self.after(100, self._poll)

    def _on_close(self) -> None:
        try:
            self._save_settings_now()
            self.engine.close()
        finally:
            self.destroy()


if __name__ == "__main__":
    AutoDJApp().mainloop()
