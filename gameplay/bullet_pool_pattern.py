"""弹幕对象池与轨迹重放。

本模块对应六类缺陷的修复：

1. ``pool_reset``     从池中取出对象时必须完整复位（速度、轨迹种子、年龄、
                      回调计数、退休/回收标记与句柄），不能继承上一轮生命周期的残留。
2. ``delayed_release``生命周期结束（retire）时若仍有在途碰撞回调持有该对象，
                      回收必须延后到回调全部退出之后。
3. ``seq_sort``       快照 / 重放必须按发射序号（spawn_seq）排序，而不是按对象
                      池槽位索引排序。
4. ``preview_clock``  编辑器预览与运行时共用同一套固定逻辑帧换算，预览不再按
                      墙钟直接推进。
5. ``pause_freeze``   暂停期间回收线程只做冻结，不做回收；恢复后统一释放。
6. ``grow_swap``      扩容采用换表：新表接管对象，旧表就地清空并作废，旧句柄
                      全部失效，不会出现同一对象被两份数组同时引用。

另外提供 ``spawn`` / ``collide`` / ``recycle`` / ``replay`` / ``determinism`` /
``validation`` / ``audit`` / ``fallback`` / ``metrics`` 等支撑入口。

所有公开函数都返回非 ``None`` 的结构。第一个参数既可以是 ``BulletPool``，
也可以是一个整数容量（此时会创建一个新的池并跑一遍最小的自检场景）。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LOGIC_HZ = 60
FRAME_SECONDS = 1.0 / LOGIC_HZ
DEFAULT_PATTERN_SEED = 0x5EEDC0DE
_MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# 确定性随机源
# ---------------------------------------------------------------------------

def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def trajectory_seed_for(pattern_seed: int, spawn_seq: int) -> int:
    """由（弹幕图案种子, 发射序号）导出轨迹种子，与对象池槽位无关。"""
    mixed = (int(pattern_seed) & _MASK64) ^ ((int(spawn_seq) & _MASK64) << 17)
    return _splitmix64(mixed)


def _velocity_for(
    pattern_seed: int, spawn_seq: int, speed: float, angle_deg: float
) -> Tuple[int, float, float, float]:
    """返回 ``(轨迹种子, 实际角度, vx, vy)``。

    结果只依赖图案种子、发射序号与发射参数，因此同一发弹幕无论落在哪个
    池槽位、被谁先取出，轨迹都一致；重放时可以直接由发射记录重建。
    """
    seed = trajectory_seed_for(pattern_seed, spawn_seq)
    jitter = ((seed >> 8) % 1001) / 1000.0 - 0.5  # [-0.5, 0.5]
    angle = math.radians(float(angle_deg)) + jitter * math.radians(10.0)
    return (
        seed,
        angle,
        float(speed) * math.cos(angle),
        float(speed) * math.sin(angle),
    )


# ---------------------------------------------------------------------------
# 逻辑帧时钟：预览与运行时共用
# ---------------------------------------------------------------------------

class LogicClock:
    """把墙钟时间换算成固定逻辑帧。

    预览面板调用 ``advance_seconds`` 得到帧数，再交给与运行时完全相同的
    ``_step_frames``；不足一帧的余量留在累加器里，不会丢失，也不会因为墙钟
    抖动而多走一帧。
    """

    __slots__ = ("hz", "frame", "_pending_seconds")

    def __init__(self, hz: int = LOGIC_HZ) -> None:
        if hz <= 0:
            raise ValueError("hz must be positive")
        self.hz = int(hz)
        self.frame = 0
        self._pending_seconds = 0.0

    @property
    def frame_seconds(self) -> float:
        return 1.0 / self.hz

    def advance_seconds(self, seconds: float) -> int:
        if seconds < 0:
            raise ValueError("seconds must be non-negative")
        self._pending_seconds += float(seconds)
        frames = int(math.floor(self._pending_seconds / self.frame_seconds + 1e-9))
        if frames:
            self._pending_seconds -= frames * self.frame_seconds
            if self._pending_seconds < 0:
                self._pending_seconds = 0.0
            self.frame += frames
        return frames

    def advance_frames(self, frames: int) -> int:
        frames = int(frames)
        if frames < 0:
            raise ValueError("frames must be non-negative")
        self.frame += frames
        return frames


# ---------------------------------------------------------------------------
# 池对象
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Handle:
    """池外持有的引用。``table_id`` 与 ``life_id`` 让扩容 / 复用后的旧引用失效。"""

    table_id: int
    slot: int
    spawn_seq: int
    life_id: int


@dataclass(frozen=True)
class SpawnRecord:
    """发射时的不可变参数，回收后仍保留在 ``spawn_log`` 中供确定性重放。"""

    spawn_seq: int
    pattern_seed: int
    speed: float
    angle_deg: float


@dataclass(eq=False)
class Bullet:
    """池中复用的弹幕对象。

    所有字段在 ``_acquire`` 中都会被显式赋值——这是缺陷 1 的修复点，不能只
    改速度而漏掉轨迹种子 / 年龄 / 回调计数等上一轮残留。
    """

    spawn_seq: int = -1
    pattern_seed: int = 0
    speed: float = 0.0
    angle_deg: float = 0.0
    angle_rad: float = 0.0
    trajectory_seed: int = 0
    vx: float = 0.0
    vy: float = 0.0
    age_frames: int = 0
    pos_x: float = 0.0
    pos_y: float = 0.0
    in_flight: int = 0
    retired: bool = False
    recycled: bool = True
    slot: int = -1
    life_id: int = 0
    handle: Optional[Handle] = None


class BulletPool:
    """固定逻辑帧驱动的弹幕对象池。"""

    def __init__(
        self, capacity: int = 1, pattern_seed: int = DEFAULT_PATTERN_SEED
    ) -> None:
        capacity = max(1, int(capacity))
        self.pattern_seed = int(pattern_seed) & _MASK64
        self.table_id = 0
        self.slots: List[Optional[Bullet]] = [None] * capacity
        self.free_slots: List[int] = list(range(capacity))
        self.free_objects: List[Bullet] = []
        self.active: List[Bullet] = []
        self.pending: List[Bullet] = []          # 已结束生命周期、等待回收
        self.spawn_log: List[SpawnRecord] = []
        self.spawn_seq = 0
        self.paused = False
        self.clock = LogicClock(LOGIC_HZ)
        self.logical_frame = 0
        self.recycled_total = 0
        self.grow_count = 0
        self.archive: List[List[Optional[Bullet]]] = []  # 就地清空后的旧表

    @property
    def capacity(self) -> int:
        return len(self.slots)

    # -- 句柄解析 ---------------------------------------------------------
    def resolve(self, handle: Optional[Handle]) -> Optional[Bullet]:
        """只有表号与生命号都对得上的句柄才能取到对象。"""
        if handle is None:
            return None
        if handle.table_id != self.table_id:
            return None
        if not (0 <= handle.slot < len(self.slots)):
            return None
        bullet = self.slots[handle.slot]
        if bullet is None:
            return None
        if bullet.spawn_seq != handle.spawn_seq or bullet.life_id != handle.life_id:
            return None
        if bullet.recycled:
            return None
        return bullet


# ---------------------------------------------------------------------------
# 内部：取出 / 结束 / 回收 / 扩容
# ---------------------------------------------------------------------------

def _coerce_pool(value: Any) -> BulletPool:
    if isinstance(value, BulletPool):
        return value
    return BulletPool(int(value))


def _is_capacity(value: Any) -> bool:
    return not isinstance(value, BulletPool)


def _acquire(pool: BulletPool, record: SpawnRecord) -> Bullet:
    """从池中取出一发弹幕。

    关键：所有每发状态都从 ``record`` 重新派生并覆盖，绝不读取对象旧值。
    """
    if not pool.free_slots:
        _grow_swap(pool, max(2, pool.capacity * 2))

    slot = pool.free_slots.pop(0)
    bullet = pool.free_objects.pop() if pool.free_objects else Bullet()
    bullet.life_id += 1

    seed, angle_rad, vx, vy = _velocity_for(
        record.pattern_seed, record.spawn_seq, record.speed, record.angle_deg
    )

    # —— 完整复位（缺陷 1）——
    bullet.spawn_seq = record.spawn_seq
    bullet.pattern_seed = record.pattern_seed
    bullet.speed = record.speed
    bullet.angle_deg = record.angle_deg
    bullet.angle_rad = angle_rad
    bullet.trajectory_seed = seed
    bullet.vx = vx
    bullet.vy = vy
    bullet.age_frames = 0
    bullet.pos_x = 0.0
    bullet.pos_y = 0.0
    bullet.in_flight = 0
    bullet.retired = False
    bullet.recycled = False
    bullet.slot = slot
    # —— 复位结束 ——

    bullet.handle = Handle(pool.table_id, slot, record.spawn_seq, bullet.life_id)
    pool.slots[slot] = bullet
    pool.active.append(bullet)
    return bullet


def _retire(pool: BulletPool, bullet: Bullet) -> bool:
    """结束一发弹幕的生命周期；若有在途回调则只挂起，不回收。"""
    if bullet.recycled:
        return False
    if not bullet.retired:
        bullet.retired = True
        if bullet in pool.active:
            pool.active.remove(bullet)
        pool.pending.append(bullet)
    # 回调未退出 / 暂停中：留在 pending，回收延后。
    if bullet.in_flight == 0 and not pool.paused:
        return _recycle_if_ready(pool, bullet)
    return False


def _recycle_if_ready(pool: BulletPool, bullet: Bullet) -> bool:
    """只有在「已结束 + 无在途回调 + 未暂停」三者都满足时才回收。"""
    if bullet.recycled or bullet.in_flight > 0 or pool.paused or not bullet.retired:
        return False

    if bullet in pool.pending:
        pool.pending.remove(bullet)

    had_table_slot = False
    if 0 <= bullet.slot < len(pool.slots) and pool.slots[bullet.slot] is bullet:
        pool.slots[bullet.slot] = None
        had_table_slot = True
    bullet.recycled = True
    bullet.retired = False
    bullet.in_flight = 0
    bullet.handle = None
    # 故意保留旧速度 / 轨迹种子等字段直到下次取用，以便“取出必须完整复位”
    # 的可验证性；真正的复位发生在 _acquire。
    # 只有确实从当前表里摘下来的槽位才归还空闲表；扩容后不在表内的待回收
    # 对象，其旧槽位可能已被别人占用，不能重复塞回 free_slots。
    if had_table_slot:
        pool.free_slots.append(bullet.slot)
    bullet.slot = -1
    pool.free_objects.append(bullet)
    pool.recycled_total += 1
    return True


def _drain_pending(pool: BulletPool) -> int:
    """回收线程的入口：暂停时只冻结，不释放。"""
    if pool.paused:
        return 0
    released = 0
    for bullet in list(pool.pending):
        if _recycle_if_ready(pool, bullet):
            released += 1
    return released


def _set_paused(pool: BulletPool, paused: bool) -> None:
    pool.paused = bool(paused)


def _grow_swap(pool: BulletPool, new_capacity: int) -> Dict[str, Any]:
    """换表扩容：对象只交给新表，旧表就地清空，旧句柄全部作废。

    不做 ``old_array = new_array`` 式的引用拷贝——那样同一个对象会被两份
    数组同时引用，回收一次后另一份仍指向它（缺陷 6）。
    """
    old_table = pool.slots
    old_table_id = pool.table_id
    live = [b for b in old_table if b is not None]
    new_capacity = max(int(new_capacity), len(live), 1)

    retired: List[Handle] = [b.handle for b in live if b.handle is not None]

    pool.table_id += 1
    new_table: List[Optional[Bullet]] = [None] * new_capacity
    for index, bullet in enumerate(live):
        bullet.slot = index
        bullet.handle = Handle(pool.table_id, index, bullet.spawn_seq, bullet.life_id)
        new_table[index] = bullet

    # 旧表立即清空，避免同一个对象被新旧两份数组同时引用。
    for index in range(len(old_table)):
        old_table[index] = None
    pool.archive.append(old_table)

    pool.slots = new_table
    pool.free_slots = list(range(len(live), new_capacity))
    pool.grow_count += 1

    return {
        "old_table_id": old_table_id,
        "new_table_id": pool.table_id,
        "migrated": len(live),
        "invalidated_handles": len(retired),
        "old_table_emptied": all(item is None for item in old_table),
        "duplicate_references": _duplicate_reference_count(pool),
    }


def _duplicate_reference_count(pool: BulletPool) -> int:
    ids = [id(b) for b in pool.slots if b is not None]
    return len(ids) - len(set(ids))


def _step_frames(pool: BulletPool, frames: int) -> int:
    """运行时与编辑器预览共用的唯一推进函数。"""
    frames = int(frames)
    if frames < 0:
        raise ValueError("frames must be non-negative")
    if frames == 0 or pool.paused:
        return 0
    dt = FRAME_SECONDS
    for bullet in pool.active:
        bullet.age_frames += frames
        bullet.pos_x += bullet.vx * dt * frames
        bullet.pos_y += bullet.vy * dt * frames
    pool.logical_frame += frames
    return frames


def _ordered_live(pool: BulletPool) -> List[Bullet]:
    """按发射序号排序，绝不按槽位索引排序（缺陷 3）。"""
    return sorted(pool.active, key=lambda b: b.spawn_seq)


def _snapshot_by_seq(pool: BulletPool, digits: int = 6) -> List[Dict[str, Any]]:
    return [
        {
            "spawn_seq": b.spawn_seq,
            "slot": b.slot,
            "x": round(b.pos_x, digits),
            "y": round(b.pos_y, digits),
            "age": b.age_frames,
        }
        for b in _ordered_live(pool)
    ]


def _record_of(bullet: Bullet) -> SpawnRecord:
    return SpawnRecord(
        bullet.spawn_seq, bullet.pattern_seed, bullet.speed, bullet.angle_deg
    )


def _twin_pool(source: BulletPool) -> BulletPool:
    """用同样的发射记录构造一个槽位排布可能不同的对照池。"""
    twin = BulletPool(
        max(1, source.capacity), pattern_seed=source.pattern_seed
    )
    records = sorted(
        (_record_of(b) for b in source.active), key=lambda r: r.spawn_seq
    )
    for record in records:
        _acquire(twin, record)
    twin.spawn_seq = max(
        [r.spawn_seq for r in source.spawn_log] + [r.spawn_seq for r in records] + [0]
    )
    twin.spawn_log = list(source.spawn_log)
    return twin


def _replay_record(record: SpawnRecord, frames: int) -> Dict[str, Any]:
    """纯函数重放：只用发射记录，不接触任何池槽位。"""
    seed, angle_rad, vx, vy = _velocity_for(
        record.pattern_seed, record.spawn_seq, record.speed, record.angle_deg
    )
    total = max(0, int(frames))
    sample_frames = sorted({0, total // 2, total})
    samples = []
    for frame in sample_frames:
        samples.append(
            {
                "frame": frame,
                "x": round(vx * FRAME_SECONDS * frame, 6),
                "y": round(vy * FRAME_SECONDS * frame, 6),
            }
        )
    return {
        "spawn_seq": record.spawn_seq,
        "trajectory_seed": seed,
        "angle_rad": round(angle_rad, 9),
        "vx": round(vx, 9),
        "vy": round(vy, 9),
        "samples": samples,
    }


def _replay_tracks(pool: BulletPool, frames: int = 30) -> List[Dict[str, Any]]:
    ordered = sorted(pool.spawn_log, key=lambda r: r.spawn_seq)
    return [_replay_record(record, frames) for record in ordered]


def _replay_digest(pool: BulletPool, frames: int = 30) -> str:
    payload = json.dumps(_replay_tracks(pool, frames), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 公开入口 1：取出时完整复位
# ---------------------------------------------------------------------------

def pool_reset(pool_or_capacity: Any) -> Dict[str, Any]:
    pool = _coerce_pool(pool_or_capacity)

    first = spawn(pool, 1, speed=10.0, angle_deg=0.0)[0]
    stale_source = pool.free_objects[-1] if pool.free_objects else first
    stale_before = {
        "vx": stale_source.vx,
        "vy": stale_source.vy,
        "trajectory_seed": stale_source.trajectory_seed,
        "age_frames": stale_source.age_frames,
        "in_flight": stale_source.in_flight,
    }
    _retire(pool, first)  # 生命结束；无在途回调，立即回收

    second = spawn(pool, 1, speed=3.0, angle_deg=90.0)[0]
    expected_seed, _, expected_vx, expected_vy = _velocity_for(
        second.pattern_seed, second.spawn_seq, second.speed, second.angle_deg
    )

    checks = {
        "object_reused": second is first,
        "velocity_reset": math.isclose(second.vx, expected_vx, abs_tol=1e-12)
        and math.isclose(second.vy, expected_vy, abs_tol=1e-12),
        "trajectory_seed_recomputed": second.trajectory_seed == expected_seed,
        "age_reset": second.age_frames == 0,
        "position_reset": second.pos_x == 0.0 and second.pos_y == 0.0,
        "in_flight_reset": second.in_flight == 0,
        "flags_reset": not second.retired and not second.recycled,
        "handle_reissued": second.handle is not None
        and second.handle.spawn_seq == second.spawn_seq
        and second.handle.table_id == pool.table_id,
    }
    return {
        "ok": all(checks.values()),
        "spawn_seq": second.spawn_seq,
        "life_id": second.life_id,
        "velocity": (round(second.vx, 9), round(second.vy, 9)),
        "expected_velocity": (round(expected_vx, 9), round(expected_vy, 9)),
        "trajectory_seed": second.trajectory_seed,
        "expected_trajectory_seed": expected_seed,
        "stale_state_before_acquire": stale_before,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# 公开入口 2：回收延迟到回调结束后
# ---------------------------------------------------------------------------

def delayed_release(pool_or_capacity: Any) -> Dict[str, Any]:
    pool = _coerce_pool(pool_or_capacity)
    bullet = spawn(pool, 1, speed=8.0, angle_deg=15.0)[0]

    bullet.in_flight += 1             # 在途碰撞回调持有该对象
    _retire(pool, bullet)             # 生命周期结束
    deferred = (
        bullet in pool.pending
        and not bullet.recycled
        and bullet not in pool.free_objects
    )
    not_recycled_by_drain = _drain_pending(pool) == 0

    bullet.in_flight -= 1             # 回调退出
    released = _recycle_if_ready(pool, bullet)

    return {
        "ok": deferred and not_recycled_by_drain and released,
        "deferred_while_callback_in_flight": deferred,
        "drain_skipped_while_callback_in_flight": not_recycled_by_drain,
        "released_after_callback_exit": released,
        "pending_after_release": len(pool.pending),
        "spawn_seq": bullet.spawn_seq,
        "recycled_total": pool.recycled_total,
    }


def collide(pool_or_capacity: Any) -> Dict[str, Any]:
    """模拟一次碰撞回调：回调内结束弹幕，但回收等回调退出后才发生。"""
    pool = _coerce_pool(pool_or_capacity)
    bullet = spawn(pool, 1, speed=7.5, angle_deg=40.0)[0]

    retire_inside_callback = False
    recycled_inside_callback = False
    try:
        bullet.in_flight += 1
        _retire(pool, bullet)
        retire_inside_callback = bullet.retired
        recycled_inside_callback = _recycle_if_ready(pool, bullet)
    finally:
        bullet.in_flight -= 1
        released = _recycle_if_ready(pool, bullet)

    return {
        "ok": bullet.recycled and retire_inside_callback and not recycled_inside_callback,
        "spawn_seq": bullet.spawn_seq,
        "retired_inside_callback": retire_inside_callback,
        "recycled_inside_callback": recycled_inside_callback,
        "recycled_after_callback_exit": bullet.recycled,
        "in_flight": bullet.in_flight,
        "pending": len(pool.pending),
    }


# ---------------------------------------------------------------------------
# 公开入口 3：按发射序号排序
# ---------------------------------------------------------------------------

def seq_sort(pool_or_bullets: Any) -> List[Dict[str, Any]]:
    """返回按 ``spawn_seq`` 排序的弹幕信息。

    若传入整数，会构造一个「先发射、后回收、再复用槽位」的场景，使
    「按槽位排序」与「按发射序号排序」产生不同结果，从而可直接验证修复。
    """
    if isinstance(pool_or_bullets, BulletPool):
        return [
            {"spawn_seq": b.spawn_seq, "slot": b.slot}
            for b in _ordered_live(pool_or_bullets)
        ]

    if isinstance(pool_or_bullets, (list, tuple)):
        bullet_like = list(pool_or_bullets)
        ordered = sorted(bullet_like, key=lambda b: b.spawn_seq)
        return [
            {"spawn_seq": b.spawn_seq, "slot": getattr(b, "slot", None)}
            for b in ordered
        ]

    pool = _coerce_pool(pool_or_bullets)
    # 演示场景：槽位全部占满后结束一发，再发射一发复用它的槽位。
    # 于是“按槽位遍历”会把新发弹幕排在中间，与发射序号顺序不同。
    if pool.capacity < 4:
        _grow_swap(pool, 4)
    fill = len(pool.free_slots)
    if fill:
        spawn(pool, fill, speed=6.0, angle_deg=0.0)

    target = sorted(pool.active, key=lambda b: b.spawn_seq)[1]
    retired_seq = target.spawn_seq
    _retire(pool, target)
    reused = spawn(pool, 1, speed=6.0, angle_deg=60.0)[0]
    ordered = _ordered_live(pool)
    by_slot = sorted(pool.active, key=lambda b: b.slot)
    return {
        "ordered_by_spawn_seq": [b.spawn_seq for b in ordered],
        "would_be_by_slot": [b.spawn_seq for b in by_slot],
        "different": [b.spawn_seq for b in ordered] != [b.spawn_seq for b in by_slot],
        "slots": {b.spawn_seq: b.slot for b in ordered},
        "retired_spawn_seq": retired_seq,
        "reused_spawn_seq": reused.spawn_seq,
        "reused_slot": reused.slot,
    }


# ---------------------------------------------------------------------------
# 公开入口 4：预览与运行时共用逻辑帧
# ---------------------------------------------------------------------------

def preview_clock(pool_or_capacity: Any) -> Dict[str, Any]:
    pool = _coerce_pool(pool_or_capacity)
    if _is_capacity(pool_or_capacity):
        spawn(pool, 2, speed=5.0, angle_deg=0.0)

    twin = _twin_pool(pool)

    # 预览：墙钟切成若干块喂给时钟，不足一帧的余量保留到下一次。
    chunks = [0.25, 0.25, 0.25, 0.25]
    produced = []
    for seconds in chunks:
        frames = pool.clock.advance_seconds(seconds)
        produced.append(frames)
        _step_frames(pool, frames)

    preview_snapshot = _snapshot_by_seq(pool)
    runtime_frames = _step_frames(twin, LOGIC_HZ)
    runtime_snapshot = _snapshot_by_seq(twin)

    # 另用一个临时时钟验证亚帧余量不会丢失。
    scratch = LogicClock(LOGIC_HZ)
    subframe = [scratch.advance_seconds(1.0 / 120.0) for _ in range(2)]

    return {
        "ok": preview_snapshot == runtime_snapshot
        and sum(produced) == LOGIC_HZ
        and runtime_frames == LOGIC_HZ,
        "logic_hz": LOGIC_HZ,
        "preview_chunks_seconds": chunks,
        "frames_per_chunk": produced,
        "preview_total_frames": sum(produced),
        "runtime_total_frames": runtime_frames,
        "preview_matches_runtime": preview_snapshot == runtime_snapshot,
        "subframe_ticks_for_1_120s": subframe,
        "positions": preview_snapshot,
    }


# ---------------------------------------------------------------------------
# 公开入口 5：暂停冻结回收
# ---------------------------------------------------------------------------

def pause_freeze(pool_or_capacity: Any) -> Dict[str, Any]:
    pool = _coerce_pool(pool_or_capacity)
    if _is_capacity(pool_or_capacity):
        spawn(pool, 2, speed=4.0, angle_deg=0.0)

    _set_paused(pool, True)
    for bullet in list(pool.active):
        _retire(pool, bullet)       # 暂停中结束：只进 pending，不回收

    pending_while_paused = len(pool.pending)
    frozen_release = _drain_pending(pool)
    recycled_while_paused = pool.recycled_total
    free_slots_while_paused = len(pool.free_slots)

    _set_paused(pool, False)
    released_after_resume = _drain_pending(pool)

    return {
        "ok": frozen_release == 0
        and pending_while_paused > 0
        and released_after_resume == pending_while_paused,
        "pending_while_paused": pending_while_paused,
        "released_while_paused": frozen_release,
        "recycled_total_while_paused": recycled_while_paused,
        "free_slots_while_paused": free_slots_while_paused,
        "released_after_resume": released_after_resume,
        "pending_after_resume": len(pool.pending),
        "paused": pool.paused,
    }


# ---------------------------------------------------------------------------
# 公开入口 6：换表扩容并作废旧引用
# ---------------------------------------------------------------------------

def grow_swap(pool_or_capacity: Any) -> Dict[str, Any]:
    pool = _coerce_pool(pool_or_capacity)
    if not pool.active:
        spawn(pool, 1, speed=6.0, angle_deg=0.0)

    watched = pool.active[-1]
    old_handle = watched.handle
    old_table = pool.slots
    old_table_id = pool.table_id
    target = max(pool.capacity + 1, pool.capacity * 2)

    report = _grow_swap(pool, target)
    stale_resolution = pool.resolve(old_handle)
    fresh_resolution = pool.resolve(watched.handle)
    old_after = [item for item in old_table if item is not None]

    return {
        "ok": bool(
            report["old_table_emptied"]
            and stale_resolution is None
            and fresh_resolution is watched
            and report["duplicate_references"] == 0
            and len(old_after) == 0
        ),
        "old_table_id": old_table_id,
        "new_table_id": pool.table_id,
        "capacity_before": len(old_table),
        "capacity_after": pool.capacity,
        "migrated": report["migrated"],
        "invalidated_handles": report["invalidated_handles"],
        "old_table_emptied": report["old_table_emptied"],
        "stale_handle_rejected": stale_resolution is None,
        "fresh_handle_resolves": fresh_resolution is watched,
        "duplicate_references": report["duplicate_references"],
        "watched_spawn_seq": watched.spawn_seq,
        "new_slot": watched.slot,
    }


def fallback(pool_or_capacity: Any) -> Dict[str, Any]:
    """扩容作废旧引用后的兜底：按发射序号重新定位并换发新句柄。"""
    pool = _coerce_pool(pool_or_capacity)
    if not pool.active:
        spawn(pool, 1, speed=6.0, angle_deg=0.0)

    watched = pool.active[-1]
    old_handle = watched.handle
    old_table_id = pool.table_id
    _grow_swap(pool, max(pool.capacity + 1, pool.capacity * 2))

    stale = pool.resolve(old_handle)
    replacement = None
    if stale is None:
        for bullet in pool.active:
            if bullet.spawn_seq == old_handle.spawn_seq:
                replacement = bullet
                break
    fallback_handle = replacement.handle if replacement is not None else None

    return {
        "ok": stale is None and replacement is not None and fallback_handle is not None,
        "old_handle_stale": stale is None,
        "fallback_used": replacement is not None,
        "fallback_spawn_seq": replacement.spawn_seq if replacement else None,
        "fallback_handle": fallback_handle,
        "old_table_id": old_table_id,
        "new_table_id": pool.table_id,
        "stale_handle_rejections": 1 if stale is None else 0,
    }


# ---------------------------------------------------------------------------
# 支撑入口
# ---------------------------------------------------------------------------

def determinism(pool_or_capacity: Any) -> Dict[str, Any]:
    """同参数下，槽位复用顺序不同也必须得到相同重放轨迹。"""
    specs = [(6.0, 0.0), (5.0, 30.0), (7.0, 80.0), (4.0, 120.0)]  # (speed, angle_deg)
    seed = 0xBEEF5EED

    pool_a = BulletPool(4, pattern_seed=seed)
    for speed, angle in specs:
        spawn(pool_a, 1, speed=speed, angle_deg=angle)      # 槽位 0,1,2,3

    pool_b = BulletPool(4, pattern_seed=seed)
    b0 = spawn(pool_b, 1, speed=specs[0][0], angle_deg=specs[0][1])[0]
    b1 = spawn(pool_b, 1, speed=specs[1][0], angle_deg=specs[1][1])[0]  # 槽位 1
    _retire(pool_b, b1)                                      # 释放槽位 1
    spawn(pool_b, 1, speed=specs[2][0], angle_deg=specs[2][1])[0]       # 复用槽位 1
    _retire(pool_b, b0)                                      # 释放槽位 0
    spawn(pool_b, 1, speed=specs[3][0], angle_deg=specs[3][1])[0]       # 复用槽位 0

    digest_a = _replay_digest(pool_a, frames=45)
    digest_b = _replay_digest(pool_b, frames=45)
    tracks_a = _replay_tracks(pool_a, frames=45)
    tracks_b = _replay_tracks(pool_b, frames=45)
    replay_seqs_a = [t["spawn_seq"] for t in tracks_a]
    replay_seqs_b = [t["spawn_seq"] for t in tracks_b]
    slot_a = [b.slot for b in sorted(pool_a.active, key=lambda b: b.spawn_seq)]
    slot_b = [b.slot for b in sorted(pool_b.active, key=lambda b: b.spawn_seq)]

    return {
        "ok": digest_a == digest_b
        and replay_seqs_a == replay_seqs_b == [1, 2, 3, 4]
        and slot_a != slot_b,
        "deterministic": digest_a == digest_b,
        "digest": digest_a,
        "digest_other_schedule": digest_b,
        "replay_order_a": replay_seqs_a,
        "replay_order_b": replay_seqs_b,
        "same_replay_order": replay_seqs_a == replay_seqs_b,
        "different_slot_layout": slot_a != slot_b,
        "slot_layout_a_of_active": slot_a,
        "slot_layout_b_of_active": slot_b,
        "spawn_order": [1, 2, 3, 4],
        "sort_key": "spawn_seq",
    }


def validation(pool_or_capacity: Any) -> Dict[str, Any]:
    pool = _coerce_pool(pool_or_capacity)
    if _is_capacity(pool_or_capacity):
        spawn(pool, 2, speed=5.0, angle_deg=0.0)

    live_objects = [b for b in pool.slots if b is not None]
    live_ids = [id(b) for b in live_objects]
    duplicate_count = len(live_ids) - len(set(live_ids))
    active_ids = {id(b) for b in pool.active}
    pending_ids = {id(b) for b in pool.pending}

    seqs = [b.spawn_seq for b in live_objects]
    log_seqs = [r.spawn_seq for r in pool.spawn_log]

    checks = {
        "no_duplicate_object_references": duplicate_count == 0,
        "no_object_is_slotless": all(0 <= b.slot < pool.capacity for b in live_objects),
        "slot_points_back_to_object": all(
            pool.slots[b.slot] is b for b in live_objects
        ),
        "active_and_pending_disjoint": not (active_ids & pending_ids),
        "pending_are_retired_and_not_recycled": all(
            b.retired and not b.recycled and b.in_flight >= 0 for b in pool.pending
        ),
        "active_not_recycled": all(not b.recycled for b in pool.active),
        "spawn_seq_unique": len(seqs) == len(set(seqs)),
        "spawn_log_unique": len(log_seqs) == len(set(log_seqs)),
        "old_tables_emptied": all(
            all(item is None for item in table) for table in pool.archive
        ),
        "free_slots_in_range": all(0 <= s < pool.capacity for s in pool.free_slots),
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "duplicate_references": duplicate_count,
        "live_count": len(live_objects),
        "active_count": len(pool.active),
        "pending_count": len(pool.pending),
        "free_slots": len(pool.free_slots),
        "table_id": pool.table_id,
    }


def audit(pool_or_capacity: Any) -> Dict[str, Any]:
    pool = _coerce_pool(pool_or_capacity)
    if _is_capacity(pool_or_capacity):
        spawn(pool, 3, speed=5.0, angle_deg=0.0)
        middle = sorted(pool.active, key=lambda b: b.spawn_seq)[1]
        _retire(pool, middle)

    slots = [
        {
            "slot": index,
            "spawn_seq": bullet.spawn_seq if bullet else None,
            "life_id": bullet.life_id if bullet else None,
            "retired": bullet.retired if bullet else None,
            "in_flight": bullet.in_flight if bullet else None,
        }
        for index, bullet in enumerate(pool.slots)
    ]
    return {
        "ok": _duplicate_reference_count(pool) == 0,
        "table_id": pool.table_id,
        "capacity": pool.capacity,
        "slots": slots,
        "free_slots": list(pool.free_slots),
        "free_objects": len(pool.free_objects),
        "active_sorted_by_seq": [b.spawn_seq for b in _ordered_live(pool)],
        "active_slots": {b.spawn_seq: b.slot for b in _ordered_live(pool)},
        "pending": [b.spawn_seq for b in pool.pending],
        "pending_in_flight": sum(b.in_flight for b in pool.pending),
        "recycled_total": pool.recycled_total,
        "archived_tables": len(pool.archive),
        "archived_tables_emptied": all(
            all(item is None for item in table) for table in pool.archive
        ),
        "duplicate_references": _duplicate_reference_count(pool),
        "paused": pool.paused,
        "logical_frame": pool.logical_frame,
    }


def metrics(pool_or_capacity: Any) -> Dict[str, Any]:
    pool = _coerce_pool(pool_or_capacity)
    if _is_capacity(pool_or_capacity):
        spawn(pool, 2, speed=5.0, angle_deg=0.0)
        _retire(pool, sorted(pool.active, key=lambda b: b.spawn_seq)[0])

    return {
        "ok": True,
        "capacity": pool.capacity,
        "table_id": pool.table_id,
        "grow_count": pool.grow_count,
        "active": len(pool.active),
        "pending": len(pool.pending),
        "free_slots": len(pool.free_slots),
        "free_objects": len(pool.free_objects),
        "spawn_total": len(pool.spawn_log),
        "next_spawn_seq": pool.spawn_seq,
        "recycled_total": pool.recycled_total,
        "reused_objects": len(pool.free_objects),
        "paused": pool.paused,
        "logical_frame": pool.logical_frame,
        "clock_frame": pool.clock.frame,
        "in_flight_callbacks": sum(b.in_flight for b in pool.slots if b is not None),
    }


def spawn(
    pool_or_capacity: Any,
    count: int = 1,
    *,
    speed: float = 6.0,
    angle_deg: float = 0.0,
    pattern_seed: Optional[int] = None,
) -> List[Bullet]:
    """发射 ``count`` 发弹幕，逐发分配单调递增的 ``spawn_seq``。"""
    pool = _coerce_pool(pool_or_capacity)
    count = int(count)
    if count < 0:
        raise ValueError("count must be non-negative")

    seed = pool.pattern_seed if pattern_seed is None else int(pattern_seed) & _MASK64
    spawned: List[Bullet] = []
    for _ in range(count):
        pool.spawn_seq += 1
        record = SpawnRecord(
            spawn_seq=pool.spawn_seq,
            pattern_seed=seed,
            speed=float(speed),
            angle_deg=float(angle_deg),
        )
        pool.spawn_log.append(record)
        spawned.append(_acquire(pool, record))
    return spawned


def recycle(pool_or_capacity: Any) -> Dict[str, Any]:
    """回收线程的显式入口：暂停时冻结；否则释放所有已就绪的待回收对象。"""
    pool = _coerce_pool(pool_or_capacity)
    if _is_capacity(pool_or_capacity):
        bullet = spawn(pool, 1, speed=6.0, angle_deg=0.0)[0]
        bullet.in_flight += 1
        _retire(pool, bullet)
        bullet.in_flight -= 1

    before = len(pool.pending)
    released = _drain_pending(pool)
    return {
        "ok": released == before and not pool.paused,
        "pending_before": before,
        "released": released,
        "pending_after": len(pool.pending),
        "paused": pool.paused,
        "recycled_total": pool.recycled_total,
    }


def replay(pool_or_capacity: Any, frames: int = 30) -> Dict[str, Any]:
    """按发射序号重放；重放只依赖发射记录，不依赖当前槽位。"""
    pool = _coerce_pool(pool_or_capacity)
    if _is_capacity(pool_or_capacity):
        spawn(pool, 3, speed=5.0, angle_deg=0.0)
        spawn(pool, 1, speed=4.0, angle_deg=45.0)

    tracks = _replay_tracks(pool, frames)
    return {
        "ok": [t["spawn_seq"] for t in tracks]
        == sorted(t["spawn_seq"] for t in tracks),
        "frames": int(frames),
        "hz": LOGIC_HZ,
        "frame_seconds": FRAME_SECONDS,
        "order": "spawn_seq",
        "spawn_seqs": [t["spawn_seq"] for t in tracks],
        "tracks": tracks,
        "digest": _replay_digest(pool, frames),
    }
