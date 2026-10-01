"""
Copyright (c) 2026-, AakashAmorce.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Dynamic Auto Resume, stock MRCC only.

When stock MRCC holds the car at a stop, MRCC will not pull away until the gap to the lead has
grown to the distance of the current following setting. This shortens the setting to 1 bar for
the pull-away and puts the driver's own setting back once the gap has opened to what that
setting would hold at the current speed, so the car moves off with traffic instead of waiting
for a long gap to form first.

The ECU is worked only through the wheel's distance switches, closed loop: one tap, then wait
for CRZ_CTRL.DISTANCE_SETTING to move before the next. Every setting it passes through is one
the driver could pick on the wheel, and stock MRCC keeps the gas, the brakes and AEB.

Invariants:
- Shorter only at a standstill in HOLD, longer only on the way back: a shorter-than-chosen gap is
  never requested while moving, and the restore cannot overshoot into a shorter gap.
- Once a tap has gone out, the episode does not end until the setting reads at or longer than the
  driver's own with no tap in the last LATE_TAP_FRAMES, so a press the ECU applies late is still
  caught and undone.
- The driver wins: a physical distance press ends the episode and keeps whatever they chose.
- Bounded: unconfirmed taps are counted per drive, and past MAX_DRIVE_MISSES it stops for the drive.

In shadow mode the same decisions are made and logged, but no frame is sent; a virtual setting
stands in for the real one.
"""
from enum import IntEnum

from opendbc.car import DT_CTRL
from opendbc.car.carlog import carlog
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.mazda.values import Buttons

# CRZ_CTRL.DISTANCE_SETTING raw values: 1 lights 4 bars (longest), 4 lights 1 bar (shortest).
# DISTANCE_LESS raises the raw, DISTANCE_MORE lowers it (see the cluster comment in carcontroller).
LONGEST_SETTING = 1
SHORTEST_SETTING = 4

# Mazda's owner's manual gives each setting's gap at 80 km/h: long ~50 m, medium ~40 m, short ~30 m,
# extremely short ~25 m. Used as time gaps (s) for the restore point, keyed by the raw setting.
_REF_SPEED = 80. * CV.KPH_TO_MS
SETTING_TIME_GAP = {1: 50. / _REF_SPEED, 2: 40. / _REF_SPEED, 3: 30. / _REF_SPEED, 4: 25. / _REF_SPEED}
# Rough standstill gap MRCC keeps behind a stopped lead (m). An estimate to tune from logs; it only
# delays the restore, so too large is the safe direction.
STOP_GAP = 4.0

# MRCC resumes on its own if the lead leaves within 3 s of the stop. Past that it holds (HOLD) and
# waits for RES or the gas, so a shorter setting cannot make it creep forward.
HOLD_ARM_FRAMES = int(3.5 / DT_CTRL)
# The body ECU drops discrete presses sent faster than about one per 200 ms (docs/zoompilot/icbm.md).
TAP_PERIOD = 0.2  # s
# A tap counts as missed if DISTANCE_SETTING has not moved by then. CRZ_CTRL runs at 50 Hz.
CONFIRM_FRAMES = int(1.0 / DT_CTRL)
# An episode only ends this long after its last tap, so a late registration is still seen.
LATE_TAP_FRAMES = int(3.0 / DT_CTRL)
# Unconfirmed taps in a row before a move is abandoned (and retried later, for a restore).
MAX_MISSES = 3
# Unconfirmed taps per drive before it stops for the rest of the drive.
MAX_DRIVE_MISSES = 30
# Restore backstops once moving: whichever comes first, along with the gap check.
RESTORE_SPEED = 20. * CV.MPH_TO_MS
RESTORE_TIMEOUT_FRAMES = int(15. / DT_CTRL)
NO_LEAD_FRAMES = int(1. / DT_CTRL)
# Stay short through a crawl: the gap check waits this long after each pull-away, so stop-and-go
# traffic does not cycle the setting at every start.
MIN_SHORT_MOVING_FRAMES = int(3. / DT_CTRL)
# A restore that keeps missing waits and tries again: quickly at first, then rarely.
RESTORE_RETRY_FRAMES = int(2. / DT_CTRL)
RESTORE_RETRY_SLOW_FRAMES = int(30. / DT_CTRL)
QUICK_RESTORE_ATTEMPTS = 3


class DarState(IntEnum):
  IDLE = 0
  SHORTENING = 1  # held at a stop, tapping toward 1 bar
  SHORT = 2  # touched the setting; waiting to pull away, or crawling
  RESTORING = 3  # tapping back toward the driver's setting
  RESTORE_PENDING = 4  # waiting to restore: cruise is off, or between retries


def desired_gap(setting: int, v_ego: float) -> float:
  """The gap (m) MRCC holds at this setting and speed, approximately."""
  return STOP_GAP + SETTING_TIME_GAP.get(setting, SETTING_TIME_GAP[LONGEST_SETTING]) * max(v_ego, 0.)


class DynamicAutoResume:
  def __init__(self, shadow: bool = False):
    self.shadow = shadow
    self.state = DarState.IDLE
    self.user_setting = 0  # the driver's setting, put back after the pull-away
    self.hold_frames = 0
    self.moving_frames = 0
    self.no_lead_frames = 0
    self.pending_tap: int | None = None  # +1 shorter / -1 longer, awaiting DISTANCE_SETTING
    self.tap_from = 0
    self.confirm_frames = 0
    self.misses = 0
    self.drive_misses = 0
    self.confirmed_any = False
    self.tapped = False  # a tap went out this episode
    self.retry_frames = 0
    self.restore_attempts = 0
    self.frames_since_tap = LATE_TAP_FRAMES
    # One episode per stop: a stop is over only once the car has moved.
    self.stop_used = False
    # The ECU ignored every tap at a stop this drive (taps are not accepted in HOLD): stop arming.
    self.hold_taps_rejected = False
    self.disabled = False
    # Shadow mode: the steps a live run would have moved the real setting.
    self.virtual_steps = 0

  def update(self, *, engaged: bool, standstill: bool, v_ego: float, setting: int, lead_present: bool,
             lead_d: float, driver_distance: bool, resume_requested: bool, can_tap: bool) -> int | None:
    """One 100 Hz frame. Returns the distance button to send now, or None.

    engaged: stock MRCC is engaged (CRZ_CTRL.CRZ_ACTIVE) and openpilot is enabled.
    setting: CRZ_CTRL.DISTANCE_SETTING, raw.
    driver_distance: either physical distance switch is down (forged frames never reach carstate).
    resume_requested: openpilot is pressing RES; a resume is never delayed for this.
    can_tap: CRZ_BTNS is free this frame and the shared press pacing allows another tap.
    """
    valid = LONGEST_SETTING <= setting <= SHORTEST_SETTING
    eff = setting + self.virtual_steps if self.shadow else setting
    self.hold_frames = self.hold_frames + 1 if (engaged and standstill) else 0
    if not standstill:
      self.stop_used = False
    self.frames_since_tap += 1

    if driver_distance:
      if standstill:
        self.stop_used = True  # the driver set the distance at this stop; leave it alone
      if self.state != DarState.IDLE:
        self.pending_tap = None
        self._finish("driver_override", eff)
        return None

    self._check_tap(eff)
    # Nothing is concluded from an unreadable setting, or while a tap may still land.
    quiet = self.pending_tap is None and self.frames_since_tap >= LATE_TAP_FRAMES
    back_home = valid and quiet and eff <= self.user_setting  # at or longer than the driver's

    if self.state == DarState.IDLE:
      if (not self.disabled and not self.stop_used and not self.hold_taps_rejected and
          self.hold_frames >= HOLD_ARM_FRAMES and not resume_requested and valid and eff < SHORTEST_SETTING):
        self.stop_used = True
        self.user_setting = eff
        self.confirmed_any = False
        self.tapped = False
        self.retry_frames = 0
        self.restore_attempts = 0
        self._enter(DarState.SHORTENING, "arm", eff)

    elif self.state == DarState.SHORTENING:
      if not engaged:
        self._leave_shortening("disengaged", eff, DarState.RESTORE_PENDING)
      elif resume_requested or not standstill:
        # The lead is already moving: the pull-away goes first, with whatever was reached.
        self._leave_shortening("resume_first", eff, DarState.SHORT)
      elif valid and eff >= SHORTEST_SETTING:
        self._enter(DarState.SHORT, "shortened", eff)
      elif self.misses >= MAX_MISSES:
        if not self.confirmed_any:
          self.hold_taps_rejected = True
        self._leave_shortening("shorten_missed", eff, DarState.SHORT)
      else:
        return self._tap(eff, SHORTEST_SETTING, valid, can_tap)

    elif self.state == DarState.SHORT:
      if not engaged:
        self._enter(DarState.RESTORE_PENDING, "disengaged", eff)
      elif standstill:
        self.moving_frames = 0
        self.no_lead_frames = 0
      else:
        self.moving_frames += 1
        self.no_lead_frames = 0 if lead_present else self.no_lead_frames + 1
        reason = None
        if v_ego >= RESTORE_SPEED:
          reason = "speed"
        elif self.moving_frames >= RESTORE_TIMEOUT_FRAMES:
          reason = "timeout"
        elif self.no_lead_frames >= NO_LEAD_FRAMES:
          reason = "no_lead"
        elif (self.moving_frames >= MIN_SHORT_MOVING_FRAMES and lead_present and
              lead_d >= desired_gap(self.user_setting, v_ego)):
          reason = "gap"
        if reason is not None:
          self._enter(DarState.RESTORING, reason, eff)

    elif self.state == DarState.RESTORING:
      if back_home:
        self._finish("restored" if eff == self.user_setting else "restored_longer", eff)
      elif not engaged:
        # The panda refuses every button but cancel while cruise is off.
        self._enter(DarState.RESTORE_PENDING, "disengaged", eff)
      elif self.misses >= MAX_MISSES:
        self.restore_attempts += 1
        slow = self.restore_attempts >= QUICK_RESTORE_ATTEMPTS
        self.retry_frames = RESTORE_RETRY_SLOW_FRAMES if slow else RESTORE_RETRY_FRAMES
        if slow:
          carlog.error({"event": "mazda_dar", "transition": "restore_stuck", "setting": eff,
                        "user_setting": self.user_setting, "attempts": self.restore_attempts})
        self._enter(DarState.RESTORE_PENDING, "restore_missed", eff)
      elif valid and eff > self.user_setting:
        # Longer only: the way back never asks for a shorter gap than the driver's.
        return self._tap(eff, self.user_setting, valid, can_tap)

    elif self.state == DarState.RESTORE_PENDING:
      self.retry_frames = max(self.retry_frames - 1, 0)
      if back_home:
        self._finish("restored" if eff == self.user_setting else "restored_longer", eff)
      elif engaged and self.retry_frames == 0 and not self.disabled:
        self._enter(DarState.RESTORING, "retry", eff)

    return None

  def _leave_shortening(self, reason: str, eff: int, touched_state: DarState) -> None:
    # A tap that went out may still land, so only an untouched episode can end here.
    self._enter(touched_state if self.tapped else DarState.IDLE, reason, eff)

  def _check_tap(self, eff: int) -> None:
    if self.pending_tap is None:
      return
    moved = eff - self.tap_from
    if moved == self.pending_tap:
      self.pending_tap = None
      self.misses = 0
      self.confirmed_any = True
    elif moved != 0:
      # Not the single step asked for. Count it against the budget and re-read from where the
      # setting is; the next tap, if any, is chosen from that reading.
      self.pending_tap = None
      self._miss(eff, "wrong_step")
    else:
      self.confirm_frames += 1
      if self.confirm_frames >= CONFIRM_FRAMES:
        self.pending_tap = None
        self._miss(eff, "unconfirmed")

  def _miss(self, eff: int, kind: str) -> None:
    self.misses += 1
    self.drive_misses += 1
    if self.drive_misses >= MAX_DRIVE_MISSES and not self.disabled:
      self.disabled = True
      carlog.error({"event": "mazda_dar", "transition": "disabled_for_drive", "setting": eff,
                    "user_setting": self.user_setting, "last_miss": kind})

  def _tap(self, eff: int, target: int, valid: bool, can_tap: bool) -> int | None:
    # Never tap off an unreadable setting: the direction would be a guess.
    if (self.disabled or self.pending_tap is not None or not valid or not can_tap or eff == target or
        self.frames_since_tap * DT_CTRL <= TAP_PERIOD):
      return None
    direction = 1 if target > eff else -1
    self.pending_tap = direction
    self.tap_from = eff
    self.confirm_frames = 0
    self.frames_since_tap = 0
    self.tapped = True
    carlog.info({"event": "mazda_dar", "tap": "less" if direction > 0 else "more", "from": eff,
                 "target": target, "shadow": self.shadow})
    if self.shadow:
      self.virtual_steps += direction
      return None
    return Buttons.DISTANCE_LESS if direction > 0 else Buttons.DISTANCE_MORE

  def _enter(self, state: DarState, reason: str, eff: int) -> None:
    self.state = state
    self.misses = 0
    self.moving_frames = 0
    self.no_lead_frames = 0
    if state == DarState.IDLE:
      self.virtual_steps = 0
    carlog.info({"event": "mazda_dar", "transition": state.name.lower(), "reason": reason, "setting": eff,
                 "user_setting": self.user_setting, "shadow": self.shadow})

  def _finish(self, reason: str, eff: int) -> None:
    self._enter(DarState.IDLE, reason, eff)
