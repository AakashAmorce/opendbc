"""
Copyright (c) 2026-, AakashAmorce.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Dynamic Auto Resume: the shorten/restore state machine against a simulated MRCC distance
setting (prompt, late, deaf, double-stepping and lossy ECUs), its distance taps on the wire, the
DISTANCE_SETTING read, the flags that turn it on, and the controller wiring.
"""
import random

import pytest

from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.carcontroller import CarController
from opendbc.car.mazda.interface import CarInterface
from opendbc.car.mazda.dynamic_auto_resume import (CONFIRM_FRAMES, HOLD_ARM_FRAMES, LATE_TAP_FRAMES, MAX_DRIVE_MISSES,
                                                    MAX_MISSES, MIN_SHORT_MOVING_FRAMES, NO_LEAD_FRAMES,
                                                    QUICK_RESTORE_ATTEMPTS, RESTORE_RETRY_SLOW_FRAMES, RESTORE_SPEED,
                                                    RESTORE_TIMEOUT_FRAMES, SHORTEST_SETTING, TAP_PERIOD, WATCH_FRAMES,
                                                    DarState, DynamicAutoResume, desired_gap)
from opendbc.car.mazda.tests.conftest import (CRZ_BTNS, DBC_NAME, SendButtonState, car_interface, car_params, car_params_sp,
                                              frames, mazda_car_state, packer, parse_frame, step)
from opendbc.car.mazda.values import CAR, Buttons
from opendbc.sunnypilot.car.interfaces import setup_interfaces
from opendbc.sunnypilot.car.mazda.values import MazdaFlagsSP

TAP_FRAMES = int(TAP_PERIOD / DT_CTRL) + 1
LESS, MORE = Buttons.DISTANCE_LESS, Buttons.DISTANCE_MORE
SETTLE = LATE_TAP_FRAMES + 300  # long enough for any episode in these rigs to wind up


class Mrcc:
  """The distance setting as the radar keeps it: a tap moves it `step` steps, clamped to 1..4,
  `latency` frames later, unless the ECU ignores it (deaf, or a random drop)."""

  def __init__(self, setting=2, latency=3, accept=True, step=1, drop=0., rng=None):
    self.setting = setting
    self.latency = latency
    self.accept = accept
    self.step = step
    self.drop = drop
    self.rng = rng or random.Random(0)
    self.queue: list[tuple[int, int]] = []

  def tap(self, frame, button):
    if self.accept and self.rng.random() >= self.drop:
      latency = self.latency() if callable(self.latency) else self.latency
      self.queue.append((frame + latency, self.step if button == LESS else -self.step))

  def tick(self, frame):
    for item in [q for q in self.queue if q[0] <= frame]:
      self.setting = min(max(self.setting + item[1], 1), SHORTEST_SETTING)
      self.queue.remove(item)


class Rig:
  def __init__(self, shadow=False, **mrcc_kwargs):
    self.dar = DynamicAutoResume(shadow=shadow)
    self.mrcc = Mrcc(**mrcc_kwargs)
    self.frame = 0
    self.taps: list[tuple[int, int, bool]] = []  # (frame, button, standstill)

  def run(self, n=1, *, engaged=True, standstill=True, v_ego=0., lead_present=True, lead_d=6., driver_less=False,
          driver_more=False, resume_requested=False, can_tap=True, setting=None):
    for _ in range(n):
      self.mrcc.tick(self.frame)
      button = self.dar.update(engaged=engaged, standstill=standstill, v_ego=v_ego,
                               setting=self.mrcc.setting if setting is None else setting,
                               lead_present=lead_present, lead_d=lead_d, driver_less=driver_less,
                               driver_more=driver_more, resume_requested=resume_requested, can_tap=can_tap)
      if button is not None:
        self.taps.append((self.frame, button, standstill))
        self.mrcc.tap(self.frame, button)
      self.frame += 1
    return self

  def buttons(self):
    return [b for _, b, _ in self.taps]

  def held(self, n=HOLD_ARM_FRAMES + 600, **kwargs):
    """Stopped in HOLD long enough to arm and finish shortening."""
    return self.run(n, **kwargs)

  def pull_away(self, n, v_ego=3., lead_d=6., **kwargs):
    return self.run(n, standstill=False, v_ego=v_ego, lead_d=lead_d, **kwargs)

  def assert_never_shorter_while_moving(self):
    assert not [t for t in self.taps if t[1] == LESS and not t[2]], "asked for a shorter gap while moving"


class TestShortening:

  def test_waits_out_mrccs_own_resume_window(self):
    rig = Rig().run(HOLD_ARM_FRAMES - 1)
    assert rig.dar.state == DarState.IDLE and not rig.taps
    rig.run(1)
    assert rig.dar.state == DarState.SHORTENING

  def test_walks_to_one_bar_one_confirmed_tap_at_a_time(self):
    rig = Rig(setting=1).held()
    assert rig.mrcc.setting == SHORTEST_SETTING
    assert rig.buttons() == [LESS] * 3, "one tap per step, no extras"
    assert rig.dar.state == DarState.SHORT and rig.dar.user_setting == 1
    gaps = [b[0] - a[0] for a, b in zip(rig.taps, rig.taps[1:], strict=False)]
    assert all(g * DT_CTRL > TAP_PERIOD for g in gaps), "taps faster than the ECU registers them"

  def test_already_at_one_bar_never_arms(self):
    rig = Rig(setting=SHORTEST_SETTING).held()
    assert rig.dar.state == DarState.IDLE and not rig.taps

  def test_nothing_while_crz_btns_is_busy(self):
    rig = Rig().held(can_tap=False)
    assert not rig.taps and rig.dar.state == DarState.SHORTENING

  def test_never_taps_off_an_unreadable_setting(self):
    rig = Rig().run(HOLD_ARM_FRAMES + 200, setting=0)
    assert not rig.taps

  def test_a_driver_press_at_the_stop_keeps_it_from_arming(self):
    rig = Rig().run(100)
    rig.run(1, driver_more=True)
    rig.held()
    assert rig.dar.state == DarState.IDLE and not rig.taps


class TestResumeFirst:

  def test_resume_before_any_tap_ends_the_episode(self):
    rig = Rig().run(HOLD_ARM_FRAMES)
    assert rig.dar.state == DarState.SHORTENING and not rig.taps
    rig.run(1, resume_requested=True)
    assert rig.dar.state == DarState.IDLE and not rig.taps

  def test_no_tap_once_openpilot_is_resuming(self):
    rig = Rig(setting=1).run(HOLD_ARM_FRAMES + TAP_FRAMES + 5)
    sent = len(rig.taps)
    assert sent >= 1
    rig.run(200, resume_requested=True)
    assert len(rig.taps) == sent, "a distance tap would race the RES presses"

  def test_a_tap_still_in_flight_keeps_the_episode_alive(self):
    # The tap went out but has not landed when the lead moves: ending the episode here would
    # leave a step nobody restores.
    rig = Rig(setting=2, latency=20)
    while not rig.taps:
      rig.run(1)
    rig.run(1, resume_requested=True)
    assert rig.dar.state == DarState.SHORT
    rig.run(30, resume_requested=True)
    assert rig.mrcc.setting == 3
    rig.pull_away(SETTLE, v_ego=RESTORE_SPEED)
    assert rig.mrcc.setting == 2 and rig.dar.state == DarState.IDLE
    rig.assert_never_shorter_while_moving()


class TestRestore:

  def shortened(self, setting=2, **kwargs):
    rig = Rig(setting=setting, **kwargs).held()
    assert rig.dar.state == DarState.SHORT and rig.mrcc.setting == SHORTEST_SETTING
    rig.taps.clear()
    return rig

  def test_restores_once_the_gap_covers_the_drivers_setting(self):
    rig = self.shortened(setting=2)
    v = 4.
    rig.pull_away(MIN_SHORT_MOVING_FRAMES + 100, v_ego=v, lead_d=desired_gap(2, v) - 1.)
    assert rig.dar.state == DarState.SHORT and not rig.taps
    rig.pull_away(SETTLE, v_ego=v, lead_d=desired_gap(2, v) + 1.)
    assert rig.buttons() == [MORE, MORE]
    assert rig.mrcc.setting == 2 and rig.dar.state == DarState.IDLE

  def test_gap_check_waits_out_the_crawl_window(self):
    rig = self.shortened()
    rig.pull_away(MIN_SHORT_MOVING_FRAMES - 1, lead_d=100.)
    assert rig.dar.state == DarState.SHORT

  def test_stays_short_through_stop_and_go(self):
    rig = self.shortened()
    for _ in range(4):
      rig.pull_away(MIN_SHORT_MOVING_FRAMES // 2, lead_d=100.)
      rig.run(100)
    assert rig.dar.state == DarState.SHORT and not rig.taps

  @pytest.mark.parametrize("trigger", ["speed", "no_lead", "timeout"])
  def test_backstops(self, trigger):
    rig = self.shortened()
    if trigger == "speed":
      rig.pull_away(1, v_ego=RESTORE_SPEED)
    elif trigger == "no_lead":
      rig.pull_away(NO_LEAD_FRAMES, lead_present=False)
    else:
      rig.pull_away(RESTORE_TIMEOUT_FRAMES, lead_d=0.)
    assert rig.dar.state == DarState.RESTORING
    rig.pull_away(SETTLE, v_ego=RESTORE_SPEED)
    assert rig.mrcc.setting == 2 and rig.dar.state == DarState.IDLE

  def test_disengaged_while_short_restores_on_the_next_engagement(self):
    rig = self.shortened()
    rig.run(300, engaged=False)
    assert rig.dar.state == DarState.RESTORE_PENDING and not rig.taps, "the panda refuses buttons while disengaged"
    rig.pull_away(SETTLE, v_ego=RESTORE_SPEED)
    assert rig.mrcc.setting == 2 and rig.dar.state == DarState.IDLE

  def test_a_deaf_ecu_keeps_the_restore_outstanding_at_a_slow_rate(self):
    rig = self.shortened()
    rig.mrcc.accept = False
    secs = 400
    rig.pull_away(int(secs / DT_CTRL), v_ego=RESTORE_SPEED)
    assert rig.dar.shortening_disabled, "a drive where most taps miss stops shortening"
    assert rig.dar.drive_misses >= MAX_DRIVE_MISSES
    assert rig.dar.state != DarState.IDLE, "a restore that never landed must not read as done"
    slow_attempts = secs / (RESTORE_RETRY_SLOW_FRAMES * DT_CTRL) + 1
    assert len(rig.taps) <= MAX_MISSES * (QUICK_RESTORE_ATTEMPTS + slow_attempts)
    assert set(rig.buttons()) == {MORE}

  def test_an_exhausted_budget_still_restores(self):
    rig = self.shortened()
    rig.dar.shortening_disabled = True
    rig.pull_away(SETTLE, v_ego=RESTORE_SPEED)
    assert rig.mrcc.setting == 2 and rig.dar.state == DarState.IDLE
    rig.run(10).held()
    assert rig.dar.state == DarState.IDLE and rig.buttons() == [MORE, MORE], "shortened again after the budget ran out"

  def test_a_few_dropped_presses_never_trip_the_budget(self):
    rng = random.Random(7)
    rig = Rig(setting=2, drop=0.07, rng=rng)
    for _ in range(150):
      rig.held(HOLD_ARM_FRAMES + 400)
      rig.pull_away(MIN_SHORT_MOVING_FRAMES + LATE_TAP_FRAMES + 200, v_ego=RESTORE_SPEED)
    assert not rig.dar.shortening_disabled
    assert rig.dar.drive_misses > 0, "the rig should have dropped some presses"
    rig.pull_away(int(120 / DT_CTRL), v_ego=RESTORE_SPEED)
    assert rig.dar.state == DarState.IDLE and rig.mrcc.setting <= 2


class TestUnreliableEcu:
  """The ECU may apply a press late, apply it twice, or drop it. The setting must never end an
  episode shorter than the driver's, and the way back never asks for a shorter gap."""

  def test_presses_applied_after_the_confirm_window_are_caught(self):
    # Every press lands 2 s late: all three shortening taps read as missed, then land anyway.
    rig = Rig(setting=2, latency=200).held()
    assert rig.dar.state == DarState.SHORT, "taps went out, so the episode must stay open"
    rig.run(300)
    assert rig.mrcc.setting == SHORTEST_SETTING
    rig.pull_away(int(60 / DT_CTRL), v_ego=RESTORE_SPEED)
    assert rig.dar.state == DarState.IDLE and rig.mrcc.setting <= 2
    rig.assert_never_shorter_while_moving()

  def test_presses_applied_after_giving_up_in_hold_are_undone(self):
    rig = Rig(setting=2, latency=500).held()
    assert rig.dar.hold_taps_rejected and rig.dar.state == DarState.SHORT
    rig.run(600)
    assert rig.mrcc.setting == SHORTEST_SETTING, "the late presses landed"
    rig.pull_away(int(120 / DT_CTRL), v_ego=RESTORE_SPEED)
    assert rig.dar.state == DarState.IDLE and rig.mrcc.setting <= 2
    rig.assert_never_shorter_while_moving()

  def test_a_quick_pull_away_does_not_outrun_a_late_press(self):
    # 3.5 s latency, the lead leaves right after the first tap, and there is no lead to wait on.
    for user in (1, 2, 3):
      rig = Rig(setting=user, latency=350).run(HOLD_ARM_FRAMES)
      while not rig.taps:
        rig.run(1)
      rig.run(30, resume_requested=True)
      rig.pull_away(int(120 / DT_CTRL), v_ego=5., lead_present=False)
      assert rig.dar.state == DarState.IDLE and rig.mrcc.setting <= user, (user, rig.mrcc.setting)
      rig.assert_never_shorter_while_moving()

  def test_a_press_landing_after_the_episode_is_undone(self):
    # 7 s latency, past the quiet window: the episode ends, then the press lands and is caught.
    rig = Rig(setting=2, latency=700).run(HOLD_ARM_FRAMES)
    while not rig.taps:
      rig.run(1)
    rig.run(30, resume_requested=True)
    rig.pull_away(LATE_TAP_FRAMES + 60, v_ego=5., lead_present=False)
    assert rig.dar.state == DarState.IDLE and rig.dar.watch_frames > 0
    rig.pull_away(int(180 / DT_CTRL), v_ego=5., lead_present=False)
    assert rig.dar.state == DarState.IDLE and rig.mrcc.setting <= 2
    assert WATCH_FRAMES > 0
    rig.assert_never_shorter_while_moving()

  def test_a_double_stepping_ecu_cannot_ping_pong(self):
    rig = Rig(setting=3, step=2).held()
    rig.pull_away(SETTLE, v_ego=RESTORE_SPEED)
    assert rig.dar.state == DarState.IDLE
    assert rig.mrcc.setting <= 3, "ended shorter than the driver's setting"
    assert len(rig.taps) <= 3
    rig.assert_never_shorter_while_moving()

  @pytest.mark.parametrize("seed", range(25))
  def test_randomized_invariants(self, seed):
    rng = random.Random(seed)
    user = rng.choice([1, 2, 3])
    rig = Rig(setting=user, latency=lambda: rng.randint(1, 450), drop=rng.choice([0., 0.2, 0.5]), rng=rng)
    for _ in range(6):  # six stops in traffic
      rig.held(HOLD_ARM_FRAMES + rng.randint(0, 800), engaged=rng.random() > 0.05)
      rig.pull_away(rng.randint(50, 1500), v_ego=rng.uniform(0.5, 12.), lead_d=rng.uniform(3., 60.),
                    lead_present=rng.random() > 0.1)
    rig.pull_away(int(600 / DT_CTRL), v_ego=RESTORE_SPEED)
    rig.assert_never_shorter_while_moving()
    assert rig.dar.state == DarState.IDLE and rig.mrcc.setting <= user, (seed, rig.mrcc.setting, user)


class TestDriverWins:

  def test_a_press_ends_the_episode_and_keeps_the_drivers_choice(self):
    rig = TestRestore().shortened()
    rig.run(LATE_TAP_FRAMES)  # no tap of ours in flight
    rig.run(1, driver_more=True)
    rig.mrcc.setting = 3  # the driver's press lands, one step longer
    assert rig.dar.state == DarState.IDLE
    rig.pull_away(int(20 / DT_CTRL), v_ego=RESTORE_SPEED)
    assert not rig.taps and rig.mrcc.setting == 3

  def test_a_press_soon_after_our_taps_keeps_the_drivers_choice(self):
    rig = TestRestore().shortened()  # our last LESS was recent
    rig.run(1, driver_more=True)
    rig.mrcc.setting = 3
    assert rig.dar.state == DarState.GUARD
    rig.pull_away(int(20 / DT_CTRL), v_ego=RESTORE_SPEED)
    assert rig.dar.state == DarState.IDLE and not rig.taps and rig.mrcc.setting == 3

  def test_our_shorter_tap_landing_on_top_of_the_drivers_press_is_undone(self):
    # 0.8 s latency, presses applied in order. The dash reads 3 with our second shorter tap still in
    # flight; the driver presses longer once, meaning 2. Ours lands (4), then theirs (3): one short.
    rig = Rig(setting=2, latency=80).run(HOLD_ARM_FRAMES)
    while len(rig.taps) < 2:
      rig.run(1)
    second_due = rig.taps[1][0] + 80
    rig.run(10)
    assert rig.mrcc.setting == 3 and rig.dar.pending_tap == 1
    rig.run(1, driver_more=True)
    rig.mrcc.queue.append((second_due + 1, -1))  # the driver's press, applied after ours
    assert rig.dar.state == DarState.GUARD
    rig.pull_away(int(30 / DT_CTRL), v_ego=RESTORE_SPEED)
    assert rig.mrcc.setting == 2, "left shorter than the driver chose"
    assert rig.dar.state == DarState.IDLE and rig.buttons()[2:] == [MORE]
    rig.assert_never_shorter_while_moving()

  def test_one_episode_per_stop(self):
    rig = Rig().run(HOLD_ARM_FRAMES)
    assert rig.dar.state == DarState.SHORTENING and not rig.taps
    rig.run(1, driver_more=True)
    rig.held()
    assert rig.dar.state == DarState.IDLE and not rig.taps, "re-armed against the driver at the same stop"
    rig.run(10, engaged=False)  # a brake tap and re-engage at the same stop
    rig.held()
    assert rig.dar.state == DarState.IDLE and not rig.taps, "re-armed after a re-engage at the same stop"
    rig.pull_away(10)
    rig.held()
    assert rig.dar.state == DarState.SHORT, "a new stop arms again"


class TestShadow:

  def test_walks_the_same_episode_without_sending(self):
    rig = Rig(shadow=True, setting=2).held()
    assert not rig.taps and rig.mrcc.setting == 2
    assert rig.dar.state == DarState.SHORT and rig.dar.virtual_steps == 2
    rig.pull_away(SETTLE, v_ego=RESTORE_SPEED)
    assert rig.dar.state == DarState.IDLE and rig.dar.virtual_steps == 0 and not rig.taps

  def test_a_real_press_still_ends_it(self):
    rig = Rig(shadow=True).held()
    rig.mrcc.setting = 1
    rig.run(1, driver_more=True)
    assert rig.dar.state == DarState.IDLE and rig.dar.virtual_steps == 0

  def test_an_unreadable_setting_never_reads_as_restored(self):
    # MRCC off can read DISTANCE_SETTING 0; 0 plus the virtual steps must not pass for the driver's.
    rig = Rig(shadow=True, setting=2).held()
    rig.run(SETTLE, engaged=False, standstill=False, v_ego=5., setting=0)
    assert rig.dar.state == DarState.RESTORE_PENDING


def test_desired_gap_grows_with_speed_and_setting():
  assert desired_gap(1, 10.) > desired_gap(4, 10.) > desired_gap(4, 0.) > 0.
  assert desired_gap(0, 10.) == desired_gap(1, 10.), "an unknown setting restores late, not early"


def test_confirm_window_outlasts_the_tap_period():
  assert CONFIRM_FRAMES * DT_CTRL > TAP_PERIOD and LATE_TAP_FRAMES > CONFIRM_FRAMES and MAX_MISSES >= 2


# On the wire

class TestDistanceTapFrame:

  @pytest.mark.parametrize("button, less, more", [(LESS, 1, 0), (MORE, 0, 1)])
  def test_one_switch_with_its_inversion(self, button, less, more):
    v = parse_frame(CRZ_BTNS, mazdacan.create_button_cmd(CANPacker(DBC_NAME), None, 5, button)[1])
    assert (v["DISTANCE_LESS"], v["DISTANCE_LESS_INV"]) == (less, 1 - less)
    assert (v["DISTANCE_MORE"], v["DISTANCE_MORE_INV"]) == (more, 1 - more)
    assert v["CAN_OFF"] == 0 and v["RES"] == 0 and v["SET_P"] == 0 and v["SET_M"] == 0 and v["TJA_BUTTON"] == 0
    assert v["CTR"] == 6

  def test_other_buttons_unchanged(self):
    for button in (Buttons.CANCEL, Buttons.RESUME, Buttons.SET_PLUS, Buttons.SET_MINUS):
      v = parse_frame(CRZ_BTNS, mazdacan.create_button_cmd(CANPacker(DBC_NAME), None, 0, button)[1])
      assert (v["DISTANCE_LESS"], v["DISTANCE_LESS_INV"], v["DISTANCE_MORE"], v["DISTANCE_MORE_INV"]) == (0, 1, 0, 1)


class TestDistanceSettingRead:

  def _feed_crz_ctrl(self, CI, setting):
    pk = packer()
    for i in range(5):
      CI.update([(int(i * DT_CTRL * 1e9), [pk.make_can_msg("CRZ_CTRL", 0, {"DISTANCE_SETTING": setting, "CRZ_ACTIVE": 1})])])

  @pytest.mark.parametrize("setting", [1, 2, 3, 4])
  def test_stock_mrcc_reports_the_setting(self, setting):
    CI = car_interface(alpha_long=False)
    self._feed_crz_ctrl(CI, setting)
    assert CI.CS.distance_setting == setting

  def test_zero_under_openpilot_longitudinal(self):
    CI = car_interface(alpha_long=True)
    self._feed_crz_ctrl(CI, 3)
    assert CI.CS.distance_setting == 0


# Flags and the controller

def flags_for(params, alpha_long=False):
  CP = car_params(CAR.MAZDA_CX5_2022, alpha_long=alpha_long)
  CP_SP = car_params_sp(CP, CAR.MAZDA_CX5_2022, alpha_long=alpha_long)
  setup_interfaces(CarInterface, CP, CP_SP, [params])
  return CP_SP.flags & (MazdaFlagsSP.DYNAMIC_AUTO_RESUME | MazdaFlagsSP.DYNAMIC_AUTO_RESUME_SHADOW)


class TestFlags:

  def test_toggle_sets_the_flag_on_stock_mrcc(self):
    assert flags_for({"MazdaDynamicAutoResume": "1"}) == MazdaFlagsSP.DYNAMIC_AUTO_RESUME
    assert flags_for({"MazdaDynamicAutoResume": "0"}) == 0
    assert flags_for({}) == 0

  def test_shadow_wins_over_the_toggle(self):
    assert flags_for({"MazdaDynamicAutoResume": "1", "MazdaDynamicAutoResumeShadow": "1"}) == \
           MazdaFlagsSP.DYNAMIC_AUTO_RESUME_SHADOW

  def test_never_under_openpilot_longitudinal(self):
    assert flags_for({"MazdaDynamicAutoResume": "1", "MazdaDynamicAutoResumeShadow": "1"}, alpha_long=True) == 0


def dar_controller(flag, candidate=CAR.MAZDA_CX5_2022, alpha_long=False):
  CP = car_params(candidate, alpha_long=alpha_long)
  CP_SP = car_params_sp(CP, candidate, alpha_long=alpha_long)
  CP_SP.flags |= flag
  return CarController({Bus.pt: DBC_NAME}, CP, CP_SP)


def drive(cc, cs, n, **kwargs):
  """Stock MRCC engaged, openpilot enabled; returns [(frame, CRZ_BTNS payload)] sent."""
  kwargs = {"long_active": False, "enabled": True, "accel": 0., "standstill": True, "cruise_engaged": True,
            "distance_setting": 2, **kwargs}
  sent = []
  for _ in range(n):
    f = cc.frame
    _, sends = step(cc, cs, **kwargs)
    sent += [(f, d) for d in frames(sends, CRZ_BTNS)]
  return sent


def dists(sent):
  out = []
  for f, d in sent:
    v = parse_frame(CRZ_BTNS, d)
    if v["DISTANCE_LESS"] or v["DISTANCE_MORE"]:
      out.append((f, LESS if v["DISTANCE_LESS"] else MORE))
  return out


class TestController:

  @pytest.fixture
  def dar(self):
    cc = dar_controller(MazdaFlagsSP.DYNAMIC_AUTO_RESUME)
    return cc, mazda_car_state(cc.CP, cc.CP_SP)

  def test_off_without_the_flag_or_under_alpha_long(self):
    assert dar_controller(0).dar is None
    assert dar_controller(MazdaFlagsSP.DYNAMIC_AUTO_RESUME, alpha_long=True).dar is None

  @pytest.mark.parametrize("candidate", [CAR.MAZDA_CX5_2022, CAR.MAZDA_CX5])
  def test_a_hold_puts_a_distance_less_tap_on_the_bus(self, candidate):
    cc = dar_controller(MazdaFlagsSP.DYNAMIC_AUTO_RESUME, candidate=candidate)
    cs = mazda_car_state(cc.CP, cc.CP_SP)
    sent = drive(cc, cs, HOLD_ARM_FRAMES + TAP_FRAMES + 5)
    assert [b for _, b in dists(sent)] == [LESS]
    assert len(sent) == 1

  def test_shadow_sends_nothing(self):
    cc = dar_controller(MazdaFlagsSP.DYNAMIC_AUTO_RESUME_SHADOW)
    cs = mazda_car_state(cc.CP, cc.CP_SP)
    assert not drive(cc, cs, HOLD_ARM_FRAMES + 300)
    assert cc.dar.state == DarState.SHORT

  def test_no_tap_while_a_physical_button_is_down(self, dar):
    cc, cs = dar
    assert not drive(cc, cs, HOLD_ARM_FRAMES + 300, accel_button=1)
    assert cc.dar.state == DarState.SHORTENING, "armed, but held its taps for the driver's press"

  def test_not_engaged_means_no_taps(self, dar):
    cc, cs = dar
    assert not dists(drive(cc, cs, HOLD_ARM_FRAMES + 300, enabled=False))
    assert cc.dar.state == DarState.IDLE

  @pytest.mark.parametrize("which", ["distance_button", "distance_more_button"])
  def test_either_physical_distance_switch_ends_the_episode(self, dar, which):
    cc, cs = dar
    drive(cc, cs, HOLD_ARM_FRAMES)
    assert cc.dar.state == DarState.SHORTENING
    drive(cc, cs, 1, **{which: 1})
    assert cc.dar.state == DarState.IDLE
    assert not dists(drive(cc, cs, 600, distance_setting=3))

  def test_resume_goes_out_and_no_distance_tap_rides_along(self, dar):
    cc, cs = dar
    drive(cc, cs, HOLD_ARM_FRAMES)
    sent = drive(cc, cs, 200, resume=True)
    assert sent and not dists(sent)
    assert all(parse_frame(CRZ_BTNS, d)["RES"] for _, d in sent)
    assert cc.dar.state == DarState.IDLE, "the untouched episode should end for the resume"

  def test_a_restore_at_a_standstill_waits_out_the_res_presses(self, dar):
    # Shortened, cruise dropped, then re-engaged at a stop while openpilot is pressing RES: the
    # restore must not slot distance taps into the RES stream.
    cc, cs = dar
    drive(cc, cs, HOLD_ARM_FRAMES)
    drive(cc, cs, 100, distance_setting=SHORTEST_SETTING)
    drive(cc, cs, 10, cruise_engaged=False, distance_setting=SHORTEST_SETTING)
    assert cc.dar.state == DarState.RESTORE_PENDING
    sent = drive(cc, cs, 300, resume=True, distance_setting=SHORTEST_SETTING)
    assert cc.dar.state == DarState.RESTORING
    assert sent and not dists(sent)
    assert [b for _, b in dists(drive(cc, cs, 50, distance_setting=SHORTEST_SETTING))] == [MORE]

  def test_the_restore_follows_the_lead_inputs(self, dar):
    cc, cs = dar
    drive(cc, cs, HOLD_ARM_FRAMES)
    drive(cc, cs, 100, distance_setting=SHORTEST_SETTING)  # the shortening landed
    assert cc.dar.state == DarState.SHORT
    v = 4.
    far, close = desired_gap(2, v) + 2., desired_gap(2, v) - 2.
    moving = {"standstill": False, "v_ego": v, "distance_setting": SHORTEST_SETTING}
    assert not dists(drive(cc, cs, MIN_SHORT_MOVING_FRAMES + 100, lead_d_rel=close, **moving)), "restored into a short gap"
    assert [b for _, b in dists(drive(cc, cs, 100, lead_d_rel=far, **moving))] == [MORE]

  def test_a_lost_lead_restores_after_a_second(self, dar):
    cc, cs = dar
    drive(cc, cs, HOLD_ARM_FRAMES)
    drive(cc, cs, 100, distance_setting=SHORTEST_SETTING)
    moving = {"standstill": False, "v_ego": 3., "distance_setting": SHORTEST_SETTING, "lead_d_rel": 5.}
    assert not dists(drive(cc, cs, NO_LEAD_FRAMES - 10, **moving))
    assert not dists(drive(cc, cs, NO_LEAD_FRAMES - 10, lead_status=False, **moving))
    assert [b for _, b in dists(drive(cc, cs, 30, lead_status=False, **moving))] == [MORE]

  def test_icbm_and_distance_taps_share_the_pacing(self, dar):
    cc, cs = dar
    sent = drive(cc, cs, HOLD_ARM_FRAMES + 300, send_button=SendButtonState.increase)
    assert any(parse_frame(CRZ_BTNS, d)["SET_P"] for _, d in sent), "ICBM kept running"
    assert dists(sent), "the distance tap still went out"
    gaps = [b[0] - a[0] for a, b in zip(sent, sent[1:], strict=False)]
    assert all(g * DT_CTRL > TAP_PERIOD for g in gaps), "two synthesized presses inside the ECU's floor"
