from __future__ import annotations

import tkinter as tk
from typing import Any, Callable

import numpy as np


def _time_text(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    value = max(0, int(seconds))
    return f"{value // 60:02d}:{value % 60:02d}"


class DJTimeline(tk.Canvas):
    """双轨波形时间轴。

    上方 A 轨支持鼠标点击和拖动预览；松开鼠标后调用 ``on_seek``。
    只在松开时真正 seek，避免拖动时反复重建切歌规划和产生音频抖动。
    """

    def __init__(
        self,
        master: tk.Misc,
        on_seek: Callable[[float], None] | None = None,
        **kwargs: Any,
    ) -> None:
        height = int(kwargs.pop("height", 142))
        super().__init__(
            master,
            height=height,
            bg="#202328",
            highlightthickness=0,
            **kwargs,
        )
        self._on_seek = on_seek
        self._last_status: dict[str, Any] = {}
        self._dragging = False
        self._preview_seconds: float | None = None
        self.bind("<Configure>", lambda _: self.event_generate("<<TimelineResize>>"))
        self.bind("<Button-1>", self._seek_press)
        self.bind("<B1-Motion>", self._seek_motion)
        self.bind("<ButtonRelease-1>", self._seek_release)

    def set_seek_callback(self, callback: Callable[[float], None] | None) -> None:
        self._on_seek = callback

    def _seek_geometry(self) -> tuple[float, float, float, float, float]:
        width = max(200, self.winfo_width())
        height = max(132, self.winfo_height())
        left = 58.0
        right = width - 12.0
        top_a = 12.0
        bottom_a = height / 2.0 - 6.0
        return left, right, top_a, bottom_a, max(1.0, right - left)

    def _time_from_pointer(self, x: float, y: float) -> float | None:
        duration = float(self._last_status.get("duration") or 0.0)
        if duration <= 0 or not self._last_status.get("current"):
            return None
        left, right, top_a, bottom_a, width = self._seek_geometry()
        if y < top_a - 8 or y > bottom_a + 8:
            return None
        ratio = float(np.clip((x - left) / width, 0.0, 1.0))
        return ratio * duration

    def _seek_press(self, event: tk.Event[tk.Misc]) -> str | None:
        value = self._time_from_pointer(float(event.x), float(event.y))
        if value is None:
            return None
        self._dragging = True
        self._preview_seconds = value
        self.render(self._last_status)
        return "break"

    def _seek_motion(self, event: tk.Event[tk.Misc]) -> str | None:
        if not self._dragging:
            return None
        value = self._time_from_pointer(float(event.x), float(event.y))
        if value is not None:
            self._preview_seconds = value
            self.render(self._last_status)
        return "break"

    def _seek_release(self, event: tk.Event[tk.Misc]) -> str | None:
        if not self._dragging:
            return None
        value = self._time_from_pointer(float(event.x), float(event.y))
        if value is None:
            value = self._preview_seconds
        self._dragging = False
        self._preview_seconds = None
        self.render(self._last_status)
        if value is not None and self._on_seek is not None:
            self._on_seek(float(value))
        return "break"

    @staticmethod
    def _x(time_value: float, duration: float, left: float, width: float) -> float:
        if duration <= 0:
            return left
        return left + np.clip(time_value / duration, 0.0, 1.0) * width

    def _draw_waveform(
        self,
        envelope: np.ndarray | None,
        top: float,
        bottom: float,
        left: float,
        right: float,
        color: str,
    ) -> None:
        center = (top + bottom) / 2.0
        half = max(1.0, (bottom - top) * 0.40)
        self.create_line(left, center, right, center, fill="#343a40")
        if envelope is None or len(envelope) < 2:
            return
        values = np.asarray(envelope, dtype=np.float64)
        points: list[float] = []
        for index, value in enumerate(values):
            x = left + index / max(len(values) - 1, 1) * (right - left)
            y = center - float(np.clip(value, 0.0, 1.0)) * half
            points.extend((x, y))
        for index in range(len(values) - 1, -1, -1):
            value = values[index]
            x = left + index / max(len(values) - 1, 1) * (right - left)
            y = center + float(np.clip(value, 0.0, 1.0)) * half
            points.extend((x, y))
        if len(points) >= 6:
            self.create_polygon(points, fill=color, outline="")

    def _marker(
        self,
        x: float,
        top: float,
        bottom: float,
        label: str,
        color: str,
        anchor: str = "nw",
    ) -> None:
        self.create_line(x, top, x, bottom, fill=color, width=2)
        dx = 4 if anchor == "nw" else -4
        self.create_text(
            x + dx,
            top + 2,
            text=label,
            fill=color,
            font=("Arial", 9, "bold"),
            anchor=anchor,
        )

    def _region(
        self,
        start: float | None,
        end: float | None,
        duration: float,
        top: float,
        bottom: float,
        left: float,
        width: float,
        fill: str,
    ) -> None:
        if start is None or end is None or duration <= 0 or end <= start:
            return
        x0 = self._x(float(start), duration, left, width)
        x1 = self._x(float(end), duration, left, width)
        self.create_rectangle(x0, top, x1, bottom, fill=fill, outline="")

    def _cue_ticks(
        self,
        cues: object,
        duration: float,
        top: float,
        bottom: float,
        left: float,
        width: float,
        color: str,
    ) -> None:
        if duration <= 0 or not isinstance(cues, (tuple, list, np.ndarray)):
            return
        for value in list(cues)[:48]:
            try:
                x = self._x(float(value), duration, left, width)
            except (TypeError, ValueError):
                continue
            self.create_line(x, bottom - 7, x, bottom, fill=color, width=1)

    def _structure_segments(
        self,
        segments: object,
        duration: float,
        top: float,
        bottom: float,
        left: float,
        width: float,
    ) -> None:
        if duration <= 0 or not isinstance(segments, (tuple, list)):
            return
        palette = {
            "INTRO": "#24435a",
            "VERSE": "#3a3154",
            "CHORUS": "#5a3a2d",
            "BREAK": "#284b3b",
            "BRIDGE": "#4d3d24",
            "INST": "#30404d",
            "SOLO": "#54334d",
            "OUTRO": "#4a2f38",
            "START": "#2d3945",
            "END": "#3f3035",
        }
        strip_bottom = min(bottom, top + 15.0)
        for item in segments:
            if not isinstance(item, (tuple, list)) or len(item) < 3:
                continue
            try:
                start, end, label = float(item[0]), float(item[1]), str(item[2]).upper()
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            x0 = self._x(start, duration, left, width)
            x1 = self._x(end, duration, left, width)
            fill = palette.get(label, "#343a40")
            self.create_rectangle(x0, top, x1, strip_bottom, fill=fill, outline="#202328")
            if x1 - x0 >= 34:
                self.create_text(
                    x0 + 3,
                    top + 7,
                    text=label,
                    fill="#e6edf3",
                    font=("Arial", 7, "bold"),
                    anchor="w",
                )

    def render(self, status: dict[str, Any]) -> None:
        self._last_status = dict(status)
        self.delete("all")
        width = max(200, self.winfo_width())
        height = max(132, self.winfo_height())
        left = 58.0
        right = width - 12.0
        plot_width = max(1.0, right - left)
        row_gap = 12.0
        top_a = 12.0
        bottom_a = height / 2.0 - row_gap / 2.0
        top_b = height / 2.0 + row_gap / 2.0
        bottom_b = height - 12.0

        self.create_rectangle(0, 0, width, height, fill="#202328", outline="")
        self.create_text(
            10,
            (top_a + bottom_a) / 2.0,
            text="A",
            fill="#ffffff",
            font=("Arial", 12, "bold"),
            anchor="w",
        )
        self.create_text(
            10,
            (top_b + bottom_b) / 2.0,
            text="B",
            fill="#ffffff",
            font=("Arial", 12, "bold"),
            anchor="w",
        )

        duration = float(status.get("duration") or 0.0)
        next_duration = float(status.get("next_duration") or 0.0)
        transition_start = status.get("transition_start")
        transition_end = status.get("transition_end")
        next_entry = status.get("next_entry")
        next_end = status.get("next_transition_end")
        restore_start = status.get("tempo_restore_start")
        restore_end = status.get("tempo_restore_end")
        next_restore_start = status.get("next_tempo_restore_start")
        next_restore_end = status.get("next_tempo_restore_end")

        self._region(
            transition_start,
            transition_end,
            duration,
            top_a,
            bottom_a,
            left,
            plot_width,
            "#4a2933",
        )
        self._region(
            next_entry,
            next_end,
            next_duration,
            top_b,
            bottom_b,
            left,
            plot_width,
            "#293d4a",
        )
        self._region(
            restore_start,
            restore_end,
            duration,
            top_a,
            bottom_a,
            left,
            plot_width,
            "#3e3824",
        )
        self._region(
            next_restore_start,
            next_restore_end,
            next_duration,
            top_b,
            bottom_b,
            left,
            plot_width,
            "#3e3824",
        )

        self._structure_segments(
            status.get("current_sections"),
            duration, top_a, bottom_a, left, plot_width
        )
        self._structure_segments(
            status.get("next_sections"),
            next_duration, top_b, bottom_b, left, plot_width
        )

        self._draw_waveform(
            status.get("current_waveform"),
            top_a,
            bottom_a,
            left,
            right,
            "#68707a",
        )
        self._draw_waveform(
            status.get("next_waveform"),
            top_b,
            bottom_b,
            left,
            right,
            "#526c7a",
        )
        self._cue_ticks(
            status.get("current_cues"), duration, top_a, bottom_a,
            left, plot_width, "#9b8afb"
        )
        self._cue_ticks(
            status.get("next_cues"), next_duration, top_b, bottom_b,
            left, plot_width, "#79c0ff"
        )

        if transition_start is not None and duration > 0:
            x = self._x(float(transition_start), duration, left, plot_width)
            self._marker(
                x,
                top_a,
                bottom_a,
                f"OUT {_time_text(float(transition_start))}",
                "#ff6b81",
            )
        if transition_end is not None and duration > 0:
            x = self._x(float(transition_end), duration, left, plot_width)
            self._marker(x, top_a, bottom_a, "MIX END", "#ffd166", anchor="ne")
        if next_entry is not None and next_duration > 0:
            x = self._x(float(next_entry), next_duration, left, plot_width)
            self._marker(
                x,
                top_b,
                bottom_b,
                f"IN {_time_text(float(next_entry))}",
                "#66d9ef",
            )
        if next_end is not None and next_duration > 0:
            x = self._x(float(next_end), next_duration, left, plot_width)
            self._marker(x, top_b, bottom_b, "MIX END", "#ffd166", anchor="ne")

        switch_a = status.get("switch_time_a")
        switch_b = status.get("switch_time_b")
        if switch_a is not None and duration > 0:
            x = self._x(float(switch_a), duration, left, plot_width)
            self._marker(x, top_a, bottom_a, "SWAP", "#c792ea")
        if switch_b is not None and next_duration > 0:
            x = self._x(float(switch_b), next_duration, left, plot_width)
            self._marker(x, top_b, bottom_b, "SWAP", "#c792ea")

        if restore_start is not None and restore_end is not None and duration > 0:
            x0 = self._x(float(restore_start), duration, left, plot_width)
            x1 = self._x(float(restore_end), duration, left, plot_width)
            self._marker(x0, top_a, bottom_a, "TEMPO", "#f7c948")
            self._marker(
                x1,
                top_a,
                bottom_a,
                f"ORIG {float(status.get('original_bpm') or 0):.1f}",
                "#f7c948",
                anchor="ne",
            )
        if (
            next_restore_start is not None
            and next_restore_end is not None
            and next_duration > 0
        ):
            x0 = self._x(float(next_restore_start), next_duration, left, plot_width)
            x1 = self._x(float(next_restore_end), next_duration, left, plot_width)
            self._marker(x0, top_b, bottom_b, "BPM RAMP", "#f7c948")
            self._marker(
                x1,
                top_b,
                bottom_b,
                f"ORIG {float(status.get('next_original_bpm') or 0):.1f}",
                "#f7c948",
                anchor="ne",
            )

        position = float(status.get("position") or 0.0)
        if duration > 0:
            play_x = self._x(position, duration, left, plot_width)
            self.create_line(
                play_x,
                top_a,
                play_x,
                bottom_a,
                fill="#ffffff",
                width=2,
            )

        if self._preview_seconds is not None and duration > 0:
            preview_x = self._x(
                float(self._preview_seconds), duration, left, plot_width
            )
            self.create_line(
                preview_x,
                top_a,
                preview_x,
                bottom_a,
                fill="#66d9ef",
                width=2,
                dash=(4, 3),
            )
            self.create_text(
                preview_x,
                bottom_a - 4,
                text=f"跳转 {_time_text(self._preview_seconds)}",
                fill="#66d9ef",
                font=("Arial", 9, "bold"),
                anchor="se" if preview_x > width * 0.72 else "sw",
            )

        self.create_rectangle(left, top_a, right, bottom_a, outline="#343a40")
        self.create_rectangle(left, top_b, right, bottom_b, outline="#343a40")

        if not status.get("current"):
            self.create_text(
                width / 2,
                height / 2,
                text="播放后将显示论文 Cue、OUT / IN、谱切换点与 BPM 恢复区",
                fill="#8b949e",
                font=("Arial", 11),
            )
