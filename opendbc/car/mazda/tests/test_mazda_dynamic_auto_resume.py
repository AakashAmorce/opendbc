"""
Copyright (c) 2026-, AakashAmorce.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Dynamic Auto Resume: the shorten/restore state machine against a simulated MRCC distance
setting, its distance taps on the wire, the DISTANCE_SETTING read, and the flags that turn it on.
"""
import pytest

from opendbc.can import CANPacker
from opendbc.car import Bus, DT_CTRL
from opendbc.car.mazda import mazdacan
from opendbc.car.mazda.carcontroller import CarController
from opendbc.car.mazda.interface import CarInterface
from opendbc.car.mazda.dynamic_auto_resume import (CONFIRM_FRAMES, HOLD_ARM_FRAMES, MAX_MISSES, MAX_RESTORE_ATTEMPTS,
                                                    MIN_SHORT_MOVING_FRAMES, NO_LEAD_FRAMES, RESTORE_RETRY_FRAMES,
                                                    RESTORE_SPEED, RESTORE_TIMEOUT_FRAMES, SHORTEST_SETTING, TAP_PERIOD,
                                                    DarState, DynamicAutoResume, desired_gap)
from opendbc.car.mazda.tests.conftest import (CRZ_BTNS, DBC_NAME, SendButtonState, car_interface, car_params, car_params_sp,
                                              frames, mazda_car_state, packer, parse_frame, step)
from opendbc.car.mazda.values import CAR, Buttons
from opendbc.sunnypilot.car.interfaces import setup_interfaces
from opendbc.sunnypilot.car.mazda.values import MazdaFlagsSP

TAP_FRAMES = int(TAP_PERIOD / DT_CTRL) + 1
LESS, MORE = Buttons.DISTANCE_LESS, Buttons.DISTANCE_MORE


class Mrcc:
  """The distance setting as the radar keeps it: a tap moves it one step, clamped to 1..4,
  `latency` frames later, unless the ECU is ignoring taps."""

  def __init__(self, setting=2, latency=3, accept=True):
    self.setting = setting
    self.latency = latency
    self.accept = accept
    self.queue: list[tuple[int, int]] = []

  def tap(self, frame, button):
    if self.accept:
      self.queue.append((frame + self.latency, 1 if button == LESS else -1))

  def tick(self, frame):
    for item in [q for q in self.queue if q[0] <= frame]:
      self.setting = min(max(self.setting + item[1], 1), SHORTEST_SETTING)
      self.queue.remove(item)


class Rig:
  def __init__(self, shadow=False, **mrcc_kwargs):
    self.dar = DynamicAutoResume(shadow=shadow)
    self.mrcc = Mrcc(**mrcc_kwargs)
    self.frame = 0
    self.taps: list[tuple[int, int]] = []  # (frame, button)

  def run(self, n=1, *, engaged=True, standstill=True, v_ego=0., lead_present=True, lead_d=6., driver_distance=False,
          resume_requested=False, can_tap=True, setting=None):
    for _ in range(n):
      self.mrcc.tick(self.frame)
      button = self.dar.update(engaged=engaged, standstill=standstill, v_ego=v_ego,
                               setting=self.mrcc.setting if setting is None else setting,
                               lead_present=lead_present, lead_d=lead_d, driver_distance=driver_distance,
                               resume_requested=resume_requested, can_tap=can_tap)
      if button is not None:
        self.taps.append((self.frame, button))
        self.mrcc.tap(self.frame, button)
      self.frame += 1
    return self

  def buttons(self):
    return [b for _, b in self.taps]

  def held(self, n=HOLD_ARM_FRAMES + 300, **kwargs):
    """Stopped in HOLD long enough to arm and finish shortening."""
    return self.run(n, **kwargs)

  def pull_away(self, n, v_ego=3., lead_d=6., **kwargs):
    return self.run(n, standstill=False, v_ego=v_ego, lead_d=lead_d, **kwargs)


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
    rig.pull_away(int(1.5 / DT_CTRL), v_ego=RESTORE_SPEED)
    assert rig.mrcc.setting == 2 and rig.dar.state == DarState.IDLE


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
    rig.pull_away(150, v_ego=v, lead_d=desired_gap(2, v) + 1.)
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
    rig.pull_away(200, v_ego=RESTORE_SPEED)
    assert rig.mrcc.setting == 2 and rig.dar.state == DarState.IDLE

  def test_disengaged_while_short_restores_on_the_next_engagement(self):
    rig = self.shortened()
    rig.run(300, engaged=False)
    assert rig.dar.state == DarState.RESTORE_PENDING and not rig.taps, "the panda refuses buttons while disengaged"
    rig.pull_away(200, v_ego=RESTORE_SPEED)
    assert rig.mrcc.setting == 2 and rig.dar.state == DarState.IDLE

  def test_a_deaf_ecu_gets_a_bounded_number_of_tries(self):
    rig = self.shortened()
    rig.mrcc.accept = False
    rig.pull_away(MAX_RESTORE_ATTEMPTS * (MAX_MISSES * (CONFIRM_FRAMES + TAP_FRAMES) + RESTORE_RETRY_FRAMES) + 500,
                  v_ego=RESTORE_SPEED)
    assert rig.dar.state == DarState.IDLE, "gave up and logged restore_failed"
    assert len(rig.taps) == MAX_RESTORE_ATTEMPTS * MAX_MISSES


class TestDriverWins:

  def test_a_press_ends_the_episode_and_keeps_the_drivers_choice(self):
    rig = TestRestore().shortened()
    rig.mrcc.setting = 3  # the driver's own press, one step longer
    rig.run(1, driver_distance=True)
    assert rig.dar.state == DarState.IDLE
    rig.pull_away(int(20 / DT_CTRL), v_ego=RESTORE_SPEED)
    assert not rig.taps and rig.mrcc.setting == 3

  def test_one_episode_per_stop(self):
    rig = Rig().run(HOLD_ARM_FRAMES)
    assert rig.dar.state == DarState.SHORTENING and not rig.taps
    rig.run(1, driver_distance=True)
    rig.held()
    assert rig.dar.state == DarState.IDLE and not rig.taps, "re-armed against the driver at the same stop"
    rig.pull_away(10)
    rig.held()
    assert rig.dar.state == DarState.SHORT, "a new stop arms again"


class TestIgnoredInHold:

  def test_ecu_ignoring_taps_at_a_stop_disables_it_for_the_drive(self):
    rig = Rig(accept=False).held()
    assert rig.dar.hold_taps_rejected and rig.dar.state == DarState.IDLE
    assert len(rig.taps) == MAX_MISSES
    rig.pull_away(10)
    rig.held()
    assert len(rig.taps) == MAX_MISSES, "kept tapping a deaf ECU at every stop"


class TestShadow:

  def test_walks_the_same_episode_without_sending(self):
    rig = Rig(shadow=True, setting=2).held()
    assert not rig.taps and rig.mrcc.setting == 2
    assert rig.dar.state == DarState.SHORT and rig.dar.virtual_steps == 2
    assert not rig.dar.owns_buttons, "shadow must not hold ICBM off"
    rig.pull_away(200, v_ego=RESTORE_SPEED)
    assert rig.dar.state == DarState.IDLE and rig.dar.virtual_steps == 0 and not rig.taps

  def test_a_real_press_still_ends_it(self):
    rig = Rig(shadow=True).held()
    rig.mrcc.setting = 1
    rig.run(1, driver_distance=True)
    assert rig.dar.state == DarState.IDLE and rig.dar.virtual_steps == 0


def test_desired_gap_grows_with_speed_and_setting():
  assert desired_gap(1, 10.) > desired_gap(4, 10.) > desired_gap(4, 0.) > 0.
  assert desired_gap(0, 10.) == desired_gap(1, 10.), "an unknown setting restores late, not early"


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


def hold_frames(cc, cs, n, **kwargs):
  """Stock MRCC engaged and holding at a stop, openpilot enabled but not resuming."""
  sent = []
  for _ in range(n):
    _, sends = step(cc, cs, long_active=False, enabled=True, accel=0., standstill=True, cruise_engaged=True, **kwargs)
    sent += frames(sends, CRZ_BTNS)
  return sent


class TestController:

  def test_off_without_the_flag_or_under_alpha_long(self):
    assert dar_controller(0).dar is None
    assert dar_controller(MazdaFlagsSP.DYNAMIC_AUTO_RESUME, alpha_long=True).dar is None

  @pytest.mark.parametrize("candidate", [CAR.MAZDA_CX5_2022, CAR.MAZDA_CX5])
  def test_a_hold_puts_a_distance_less_tap_on_the_bus(self, candidate):
    cc = dar_controller(MazdaFlagsSP.DYNAMIC_AUTO_RESUME, candidate=candidate)
    cs = mazda_car_state(cc.CP, cc.CP_SP)
    sent = hold_frames(cc, cs, HOLD_ARM_FRAMES + TAP_FRAMES + 5, distance_setting=2)
    assert len(sent) == 1
    v = parse_frame(CRZ_BTNS, sent[0])
    assert v["DISTANCE_LESS"] == 1 and v["DISTANCE_MORE"] == 0

  def test_shadow_sends_nothing(self):
    cc = dar_controller(MazdaFlagsSP.DYNAMIC_AUTO_RESUME_SHADOW)
    cs = mazda_car_state(cc.CP, cc.CP_SP)
    assert not hold_frames(cc, cs, HOLD_ARM_FRAMES + 300, distance_setting=2)
    assert cc.dar.state == DarState.SHORT

  def test_no_tap_while_a_physical_button_is_down(self):
    cc = dar_controller(MazdaFlagsSP.DYNAMIC_AUTO_RESUME)
    cs = mazda_car_state(cc.CP, cc.CP_SP)
    assert not hold_frames(cc, cs, HOLD_ARM_FRAMES + 300, distance_setting=2, accel_button=1)

  def test_icbm_waits_while_a_tap_sequence_runs(self):
    cc = dar_controller(MazdaFlagsSP.DYNAMIC_AUTO_RESUME)
    cs = mazda_car_state(cc.CP, cc.CP_SP)
    before = hold_frames(cc, cs, HOLD_ARM_FRAMES, distance_setting=2, send_button=SendButtonState.increase)
    assert any(parse_frame(CRZ_BTNS, d)["SET_P"] for d in before), "ICBM should run until the episode arms"
    assert cc.dar.state == DarState.SHORTENING
    during = hold_frames(cc, cs, 100, distance_setting=2, send_button=SendButtonState.increase)
    assert cc.dar.state == DarState.SHORTENING
    assert during, "the distance tap should still go out"
    for dat in during:
      v = parse_frame(CRZ_BTNS, dat)
      assert v["SET_P"] == 0, "ICBM pressed SET+ in the middle of a distance sequence"

  def test_not_engaged_means_no_taps(self):
    cc = dar_controller(MazdaFlagsSP.DYNAMIC_AUTO_RESUME)
    cs = mazda_car_state(cc.CP, cc.CP_SP)
    sent = []
    for _ in range(HOLD_ARM_FRAMES + 300):
      _, sends = step(cc, cs, long_active=False, enabled=False, accel=0., standstill=True, cruise_engaged=True,
                      distance_setting=2)
      sent += frames(sends, CRZ_BTNS)
    assert not any(parse_frame(CRZ_BTNS, d)["DISTANCE_LESS"] for d in sent)
