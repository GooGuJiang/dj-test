from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from autodj.audio_engine import AutoDJEngine, EngineConfig
from autodj.beat_this_analyzer import BeatThisAnalyzer
from autodj.models import TrackAnalysis
from autodj.timeline import DJTimeline


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
        self.title("Beat This! Research Auto DJ")
        # 根据实际屏幕尺寸决定首次窗口大小，避免在 1366×768、
        # 小型笔记本或带系统缩放的屏幕上启动后超出可视范围。
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = min(1240, max(760, screen_width - 80))
        window_height = min(860, max(560, screen_height - 100))
        window_x = max(0, (screen_width - window_width) // 2)
        window_y = max(0, min(30, (screen_height - window_height) // 3))
        self.geometry(
            f"{window_width}x{window_height}+{window_x}+{window_y}"
        )
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

        self._build_style()
        self._build_ui()
        self._refresh_devices()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll)

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
        ttk.Label(header, text="Beat This! Research Auto DJ", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.header_subtitle = ttk.Label(
            header,
            text="智能 OUT/IN · BPM 自动回归 · 三段 EQ · Filter · Echo Out",
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
        self.timeline = DJTimeline(timeline_card)
        self.timeline.pack(fill=tk.X, expand=True)
        time_row = ttk.Frame(timeline_card, style="Card.TFrame")
        time_row.pack(fill=tk.X, padx=55, pady=(2, 0))
        self.time_left = ttk.Label(time_row, text="00:00", style="Muted.TLabel")
        self.time_left.pack(side=tk.LEFT)
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

        table_frame = ttk.Frame(queue_frame, style="Card.TFrame")
        table_frame.pack(fill=tk.BOTH, expand=True)
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        columns = ("index", "title", "bpm", "meter", "duration", "mix", "state")
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
            "mix": "智能切歌点",
            "state": "状态",
        }
        widths = {
            "index": 38,
            "title": 250,
            "bpm": 65,
            "meter": 55,
            "duration": 65,
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

        ttk.Label(settings_frame, text="智能混音设置", style="Now.TLabel").pack(
            anchor=tk.W, pady=(0, 14)
        )
        ttk.Label(
            settings_frame,
            text="过渡长度",
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
            text="自动模式会同时比较多个 OUT/IN 组合和长度",
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
            values=("auto", "Rubber Band R3", "librosa"),
            state="readonly",
        )
        stretch_backend_box.pack(fill=tk.X, pady=(4, 4))
        stretch_backend_box.bind(
            "<<ComboboxSelected>>", lambda _: self._apply_stretch_backend()
        )
        ttk.Label(
            settings_frame,
            text="auto 会优先调用 Rubber Band R3；未安装时回退 librosa",
            style="Muted.TLabel",
            wraplength=280,
        ).pack(anchor=tk.W, pady=(0, 10))

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
        self.compute_var = tk.StringVar(value="cpu")
        ttk.Combobox(
            settings_frame,
            textvariable=self.compute_var,
            values=("cpu", "cuda", "mps"),
            state="readonly",
        ).pack(fill=tk.X, pady=(4, 12))

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
        status_card.bind(
            "<Configure>",
            lambda event: self.status_label.configure(
                wraplength=max(220, event.width - 180)
            ),
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

    def _apply_policy(self) -> None:
        self.engine.set_automix_policy(self.policy_var.get())

    def _apply_transition_engine(self) -> None:
        self.engine.set_transition_engine(self.transition_engine_var.get())

    def _apply_stretch_backend(self) -> None:
        self.engine.set_time_stretch_backend(self.stretch_backend_var.get())

    def _set_effect_strength(self, value: str) -> None:
        number = float(value)
        self.effect_label.configure(text=f"{number:.0f}%")
        self.engine.set_effect_strength(number / 100.0)

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
        if self.analysis_active:
            messagebox.showinfo("正在分析", "请等待当前 Beat This! 分析完成。")
            return
        self.analysis_active = True

        def worker() -> None:
            try:
                analyzer = BeatThisAnalyzer(
                    checkpoint=self._checkpoint_name(),
                    device=self.compute_var.get(),
                )
                for path in paths:
                    try:
                        result = analyzer.analyze(
                            path,
                            status=lambda text: self.ui_events.put(("status", text)),
                            force=force,
                        )
                        self.ui_events.put(("track", result))
                    except Exception as exc:
                        self.ui_events.put(("error", f"{path.name}：{exc}"))
            finally:
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
                return
        self.tracks.append(track)
        self._refresh_tree()
        if len(self.tracks) == 1:
            self.tree.selection_set("0")

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
                    mix,
                    state,
                ),
            )
            if track.path == selected_path:
                self.tree.selection_set(str(index))

    def _remove_selected(self) -> None:
        if self.engine.get_status()["playing"]:
            messagebox.showinfo("播放中", "停止播放后再修改当前队列。")
            return
        selection = self.tree.selection()
        if selection:
            del self.tracks[int(selection[0])]
            self._refresh_tree()

    def _clear(self) -> None:
        self._stop()
        self.tracks.clear()
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
        device = self.device_map.get(self.output_var.get())
        self.starting_playback = True
        self.play_button.state(["disabled"])
        self._log(f"正在提取智能匹配特征：{playlist[index].title}")

        self.engine.set_auto_mix(self.auto_mix_var.get())
        self.engine.set_volume(self.volume_var.get() / 100.0)
        self.engine.set_crossfade_bars(self._bars_value())
        self.engine.set_max_stretch_percent(self.stretch_var.get())
        self.engine.set_tempo_restore_bars(self._restore_bars_value())
        self.engine.set_mix_style(self.style_var.get())
        self.engine.set_effect_strength(self.effect_var.get() / 100.0)
        self.engine.set_automix_policy(self.policy_var.get())
        self.engine.set_transition_engine(self.transition_engine_var.get())
        self.engine.set_time_stretch_backend(self.stretch_backend_var.get())

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
        self._refresh_tree()
        self._log("已停止。")

    def _next(self) -> None:
        self.engine.request_next()

    def _set_stretch(self, value: str) -> None:
        number = float(value)
        self.stretch_label.configure(text=f"±{number:.1f}%")
        self.engine.set_max_stretch_percent(number)

    def _log(self, text: str) -> None:
        self.status_label.configure(text=text)

    def _poll_ui_events(self) -> None:
        while True:
            try:
                kind, payload = self.ui_events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self._log(str(payload))
            elif kind == "track":
                self._insert_or_replace_track(payload)  # type: ignore[arg-type]
            elif kind == "error":
                self._log(str(payload))
                messagebox.showerror("Beat This! 分析失败", str(payload))
            elif kind == "analysis_done":
                self.analysis_active = False
                self._log("Beat This! 分析队列完成。")
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
        bars_text = f"{bars} 小节" if bars > 0 else "无缝裁切"
        self.plan_label.configure(
            text=(
                f"研究型切歌：A OUT {format_time(float(start))}  →  "
                f"B IN {format_time(float(entry))} · {bars_text} · "
                f"{transition_mode} / {policy_mode}{tempo_text}"
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
                f"节奏      {float(metrics.get('rhythm', 0.0)) * 100:5.0f}\n"
                f"低频净度  {float(metrics.get('bass_clean', 0.0)) * 100:5.0f}\n"
                f"人声避让  {float(metrics.get('vocal_clean', 0.0)) * 100:5.0f}\n"
                f"EDM 置信  {float(metrics.get('edm_confidence', 0.0)) * 100:5.0f}\n"
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
        next_text = "正在准备…" if status["next_loading"] else (status["next"] or "—")
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

        self.cpu_label.configure(text=f"Audio CPU {status['cpu_load'] * 100:.0f}%")
        if status["callback_status"]:
            self.cpu_label.configure(text=f"音频警告：{status['callback_status']}")

        if (
            status["index"] != self.last_engine_index
            or status["plan_signature"] != self.last_plan_signature
        ):
            self.last_engine_index = status["index"]
            self.last_plan_signature = status["plan_signature"]
            self._refresh_tree(status)

        self.play_button.configure(text="▶ 继续" if status["paused"] else "▶ 播放")
        self.after(100, self._poll)

    def _on_close(self) -> None:
        try:
            self.engine.close()
        finally:
            self.destroy()


if __name__ == "__main__":
    AutoDJApp().mainloop()
