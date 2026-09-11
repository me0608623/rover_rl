"""模型 bundle 與執行參數的嚴格一致性檢查（handoff #5）。

現況風險：載入模型只檢查 `raw_obs_dim == 83` 就開啟 action stacking，其餘全靠
yaml 寫對。若 `act_max_angular_velocity` 被寫成 0.785（v3c 已知 bug），車子照樣
正常啟動，只是**每一個轉向命令都是錯的** —— 沒有任何徵兆。

本模組在啟動時逐項比對，**不符就拒絕啟動**（raise，不是 warn）。
寧可起不來，也不要起一個安靜地做錯事的節點。

比對來源：
    bundle meta      → 觀測維度、frame_stack、logits
    obs_spec.json    → act_hist 模式與正規化分母
    action fixture   → decode 常數（19×19、v/ω/a 上限、control_dt、倒車縮放）
    yaml 執行參數    → 上述常數的實際生效值
    檔案 sha256      → 模型本體
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field


class ManifestMismatch(RuntimeError):
    """規格不符。訊息會列出所有不符項，不是只報第一個。"""


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class ManifestReport:
    ok: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def text(self) -> str:
        lines = []
        for name, good, detail in self.checks:
            lines.append(f"  [{'OK ' if good else 'FAIL'}] {name}: {detail}")
        return "\n".join(lines)


def _close(a, b, tol=1e-9) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)
    except (TypeError, ValueError):
        return False


def verify_bundle(
    *,
    model_path: str,
    bundle,
    obs_spec: dict | None,
    fixture: dict | None,
    runtime: dict,
    expected_sha256: str | None = None,
    strict: bool = True,
) -> ManifestReport:
    """逐項比對。

    Args:
        bundle:   model_runtime.load_bundle() 的回傳（讀 raw_obs_dim / frame_stack…）
        obs_spec: 模型旁的 *.obs_spec.json（None = 略過該組檢查並記為 FAIL）
        fixture:  訓練端 action contract（decode 常數的權威來源）
        runtime:  yaml 實際生效值，需含
                  act_max_linear_velocity / act_max_angular_velocity /
                  act_max_linear_accel / act_max_angular_accel /
                  control_dt / reverse_velocity_scale / speed_rate /
                  act_stack_size / act_stack_a_max / act_stack_omega_max
        strict:   True = 有任一 FAIL 就 raise
    """
    rep = ManifestReport(ok=True)

    def chk(name: str, good: bool, detail: str) -> None:
        rep.checks.append((name, bool(good), detail))
        if not good:
            rep.ok = False

    # ── 模型檔案本體 ──
    chk("model_exists", os.path.isfile(model_path), model_path)
    if expected_sha256:
        got = sha256_file(model_path) if os.path.isfile(model_path) else ""
        chk("model_sha256", got == expected_sha256,
            f"got={got[:16]}… want={expected_sha256[:16]}…")

    # ── bundle meta ──
    raw = int(getattr(bundle, "raw_obs_dim", -1))
    chk("raw_obs_dim_is_79_or_83", raw in (79, 83), str(raw))
    e2e = bool(getattr(bundle, "end_to_end", False))
    fs = int(getattr(bundle, "frame_stack", 1))
    lh = int(getattr(bundle, "lidar_hist_dim", 0))
    if e2e:
        chk("frame_stack_ge_2", fs >= 2, f"K={fs}")
        chk("lidar_hist_dim_consistent", lh == (fs - 1) * 72,
            f"{lh} vs (K-1)*72={(fs-1)*72}")
    chk("total_logits_38", int(getattr(bundle, "total_logits", -1)) == 38,
        str(getattr(bundle, "total_logits", None)))

    # ── obs_spec：act_hist 模式與正規化分母 ──
    if obs_spec is None:
        chk("obs_spec_present", False, "缺 *.obs_spec.json，無法驗 act_hist 契約")
    else:
        spec_raw = int(obs_spec.get("raw_obs_dim", -1))
        chk("obs_spec_matches_bundle", spec_raw == raw,
            f"spec={spec_raw} bundle={raw}")
        if raw == 83:
            chk("act_stack_size_2", int(runtime.get("act_stack_size", -1)) == 2,
                str(runtime.get("act_stack_size")))
            chk("act_stack_a_max",
                _close(runtime.get("act_stack_a_max"),
                       obs_spec.get("act_stack_a_max"), 1e-9),
                f"yaml={runtime.get('act_stack_a_max')} "
                f"spec={obs_spec.get('act_stack_a_max')}")
            chk("act_stack_omega_max",
                _close(runtime.get("act_stack_omega_max"),
                       obs_spec.get("act_stack_omega_max"), 1e-8),
                f"yaml={runtime.get('act_stack_omega_max')} "
                f"spec={obs_spec.get('act_stack_omega_max')}")
        if e2e:
            chk("obs_spec_frame_stack",
                int(obs_spec.get("frame_stack", -1)) == fs,
                f"spec={obs_spec.get('frame_stack')} bundle={fs}")

    # ── fixture：decode 常數 ──
    if fixture is None:
        chk("action_fixture_present", False,
            "缺 action contract fixture，decode 常數無權威來源")
    else:
        p = fixture["parameters"]
        pairs = [
            ("num_bins", 19, p["num_bins"]),
            ("act_max_linear_velocity", runtime.get("act_max_linear_velocity"),
             p["max_linear_velocity_mps"]),
            ("act_max_angular_velocity", runtime.get("act_max_angular_velocity"),
             p["max_angular_velocity_rad_s"]),
            ("act_max_linear_accel", runtime.get("act_max_linear_accel"),
             p["max_linear_accel_mps2"]),
            ("act_max_angular_accel", runtime.get("act_max_angular_accel"),
             p["max_angular_accel_rad_s2"]),
            ("control_dt", runtime.get("control_dt"), p["control_dt_s"]),
            ("reverse_velocity_scale", runtime.get("reverse_velocity_scale"),
             p["reverse_velocity_scale"]),
        ]
        for name, got, want in pairs:
            chk(name, _close(got, want, 1e-9), f"yaml={got} fixture={want}")

        sem = fixture.get("semantics", {})
        chk("reverse_bound_inside_decoder",
            sem.get("reverse_bound_location") == "inside decoder",
            str(sem.get("reverse_bound_location")))
        chk("issued_history_post_decode",
            "after decode/slew" in str(sem.get("issued_history", "")),
            str(sem.get("issued_history"))[:60])

    # ── 執行期安全旗標 ──
    sr = runtime.get("speed_rate")
    chk("speed_rate_in_range", sr is not None and 0.0 < float(sr) <= 1.0, str(sr))
    chk("cmd_passthrough_true_for_83d",
        (raw != 83) or bool(runtime.get("cmd_passthrough", False)),
        f"raw={raw} cmd_passthrough={runtime.get('cmd_passthrough')}"
        "（83D 需 passthrough 才能保證 published == issued）")

    if strict and not rep.ok:
        raise ManifestMismatch(
            "模型/參數規格不符，拒絕啟動：\n" + rep.text()
        )
    return rep


def load_json_if_exists(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
