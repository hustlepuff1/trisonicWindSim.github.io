"use strict";

// Run with: node --test test_controller_tuning.js
// Use the saved ROM payload; these tests never train or rewrite a model.
const assert = require("node:assert/strict");
const {test} = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const directory = path.join(__dirname, "model_bundle/phase5_Mach_del_L/v09_continuous_control");
const {Simulator} = require(path.join(directory, "control_engine.js"));
const payload = JSON.parse(fs.readFileSync(path.join(directory, "results/simulator_payload.json"), "utf8"));
const fresh = (mode = "mach", config = {}) => new Simulator(payload, 0, mode, config);
const near = (actual, expected, tolerance = 1e-10) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected}`);
const snapshot = sim => JSON.stringify({state: sim.getState(), tuning: sim.getTuning(),
  config: sim.config, field: sim.getFieldState(), commandField: sim.commandField});
function physicalState(sim) {
  const state = {...sim.getState()};
  for (const key of ["velocity_mm_s", "velocity_command_mm_s", "integral_velocity_mm_s",
    "position_limited", "speed_limited", "status"]) delete state[key];
  return state;
}
function drive(sim, ticks = 60) {
  const state = sim.getState();
  sim.setCommand((state.command_min + state.command_max) / 2);
  sim.step(ticks);
  return sim;
}

test("default controller reproduces every saved baseline state exactly", () => {
  const sim = fresh(), states = [sim.getState()];
  const run = count => { for (let i = 0; i < count; i++) states.push(sim.step()); };
  states.push(sim.setCommand(1.75)); run(100);
  states.push(sim.setCommand(3.25)); run(120);
  states.push(sim.setMode("position"));
  const state = sim.getState();
  states.push(sim.setCommand((state.command_min + state.command_max) / 2)); run(100);
  states.push(sim.setMode("mach")); run(10);
  // Captured from the unmodified engine before adding the tuning API; includes
  // all 336 getState() objects, including their key names and numerical values.
  const digest = crypto.createHash("sha256").update(JSON.stringify(states)).digest("hex");
  assert.equal(states.length, 336);
  assert.equal(digest, "72f423637663afd03b0bcc48b48c88cf11eff106319faa8e1ac711f3e8d8aa5b");
});

test("defaults snapshot resolved initial settings and cannot be changed through the getter", () => {
  const sim = fresh("mach", {max_speed_mm_s: 0.7, mach_kp_per_s: 1.4,
    mach_ki_per_s2: 0.3, position_kp_per_s: 2.2, position_ki_per_s2: 0.4});
  const expected = {max_speed_mm_s: 0.7, kp_per_s: 1.4, ki_per_s2: 0.3};
  assert.deepEqual(sim.getTuning(), {mode: "mach", ...expected, defaults: expected});
  sim.getTuning().defaults.kp_per_s = 999;
  sim.setTuning({max_speed_mm_s: 0.9, kp_per_s: 1.7});
  assert.deepEqual(sim.getTuning().defaults, expected);
  sim.setMode("position");
  assert.deepEqual(sim.getTuning().defaults, {max_speed_mm_s: 0.7, kp_per_s: 2.2, ki_per_s2: 0.4});
});

test("applying identical values is an exact no-op, including while moving", () => {
  for (const sim of [fresh(), drive(fresh())]) {
    const before = snapshot(sim), field = sim.field, commandField = sim.commandField;
    const {max_speed_mm_s, kp_per_s, ki_per_s2} = sim.getTuning();
    sim.setTuning({max_speed_mm_s, kp_per_s, ki_per_s2});
    sim.setTuning({});
    assert.equal(snapshot(sim), before);
    assert.equal(sim.field, field);
    assert.equal(sim.commandField, commandField);
  }
});

test("lowering speed clips both velocities immediately; raising it permits faster subsequent motion", () => {
  const sim = drive(fresh());
  assert.ok(Math.abs(sim.getState().velocity_mm_s) > 0.49);
  const before = physicalState(sim), field = sim.field, commandField = sim.commandField;
  sim.setTuning({max_speed_mm_s: 0.1});
  assert.deepEqual(physicalState(sim), before);
  assert.equal(sim.field, field);
  assert.equal(sim.commandField, commandField);
  assert.equal(sim.getState().velocity_mm_s, -0.1);
  assert.equal(sim.getState().velocity_command_mm_s, -0.1);
  assert.equal(sim.getState().speed_limited, true);
  sim.setTuning({max_speed_mm_s: 1.5});
  assert.deepEqual(physicalState(sim), before);
  assert.equal(sim.getState().velocity_mm_s, -0.1);
  assert.equal(sim.getState().velocity_command_mm_s, -0.1);
  for (let i = 0; i < 100; i++) {
    const state = sim.step();
    assert.ok(Math.abs(state.velocity_mm_s) <= 1.5);
    assert.ok(Math.abs(state.velocity_command_mm_s) <= 1.5);
  }
  assert.ok(Math.abs(sim.getState().velocity_mm_s) > 0.5);
});

test("PI gain changes preserve the previous velocity command and physical state in both modes", () => {
  for (const mode of ["mach", "position"]) {
    const sim = drive(fresh(mode));
    const before = sim.getState(), physical = physicalState(sim);
    const field = sim.field, commandField = sim.commandField;
    sim.setTuning({kp_per_s: 0.7, ki_per_s2: 0.6});
    const after = sim.getState();
    assert.deepEqual(physicalState(sim), physical);
    assert.equal(sim.field, field);
    assert.equal(sim.commandField, commandField);
    assert.equal(after.velocity_mm_s, before.velocity_mm_s);
    assert.equal(after.velocity_command_mm_s, before.velocity_command_mm_s);
    near(0.7 * after.equivalent_control_error_mm + after.integral_velocity_mm_s,
      before.velocity_command_mm_s);
    near(sim.step().velocity_command_mm_s, before.velocity_command_mm_s);
  }
});

test("mode gains persist independently while speed is common and defaults remain restorable", () => {
  const sim = fresh();
  sim.setTuning({max_speed_mm_s: 0.8, kp_per_s: 1.8, ki_per_s2: 0.4});
  sim.setMode("position");
  assert.deepEqual(sim.getTuning(), {mode: "position", max_speed_mm_s: 0.8,
    kp_per_s: 2, ki_per_s2: 0.5, defaults: {max_speed_mm_s: 0.5, kp_per_s: 2, ki_per_s2: 0.5}});
  sim.setTuning({max_speed_mm_s: 0.9, kp_per_s: 2.8, ki_per_s2: 0.8});
  sim.setMode("mach");
  assert.equal(sim.getTuning().kp_per_s, 1.8);
  assert.equal(sim.getTuning().ki_per_s2, 0.4);
  assert.equal(sim.getTuning().max_speed_mm_s, 0.9);
  sim.setTuning(sim.getTuning().defaults);
  assert.equal(sim.getTuning().kp_per_s, 1.2);
  assert.equal(sim.getTuning().max_speed_mm_s, 0.5);
  sim.setMode("position");
  assert.equal(sim.getTuning().kp_per_s, 2.8);
  assert.equal(sim.getTuning().ki_per_s2, 0.8);
  sim.setTuning(sim.getTuning().defaults);
  sim.reset();
  assert.equal(sim.getTuning().kp_per_s, 2);
  assert.equal(sim.getTuning().ki_per_s2, 0.5);
});

test("invalid multi-value transactions leave every setting and state unchanged", () => {
  const sim = drive(fresh());
  const invalid = [null, [], 1, "1", {max_speed_mm_s: 0}, {max_speed_mm_s: -1},
    {max_speed_mm_s: Infinity}, {max_speed_mm_s: "0.7"}, {kp_per_s: 0},
    {ki_per_s2: -0.1}, {ki_per_s2: NaN}, {ki_per_s2: undefined},
    {max_speed_mm_s: 0.7, kp_per_s: -2}, {max_speed_mm_s: 0.7, sample_time_s: 0.1},
    {kp_per_s: Number.MAX_VALUE}];
  for (const settings of invalid) {
    const before = snapshot(sim), field = sim.field;
    assert.throws(() => sim.setTuning(settings));
    assert.equal(snapshot(sim), before);
    assert.equal(sim.field, field);
  }
});

test("I=0 means pure P: a newly issued target generates motion and no integral accumulates", () => {
  for (const mode of ["mach", "position"]) {
    const sim = fresh(mode), initial = sim.getState();
    sim.setCommand((initial.command_min + initial.command_max) / 2);
    const physical = physicalState(sim);
    sim.setTuning({ki_per_s2: 0});
    assert.deepEqual(physicalState(sim), physical);
    assert.equal(sim.getState().integral_velocity_mm_s, 0);
    assert.equal(sim.getState().velocity_mm_s, 0);
    assert.ok(sim.getState().velocity_command_mm_s < 0);
    for (let i = 0; i < 100; i++) {
      const before = sim.getState(), kp = sim.getTuning().kp_per_s;
      const expected = Math.max(-0.5, Math.min(0.5, kp * before.equivalent_control_error_mm));
      const state = sim.step();
      near(state.velocity_command_mm_s, expected);
      assert.equal(state.integral_velocity_mm_s, 0);
    }
    assert.ok(sim.getState().actual_L_mm < initial.actual_L_mm);
  }
});

test("entering a mode whose I gain is zero clears previous PI integral", () => {
  for (const pureMode of ["mach", "position"]) {
    const otherMode = pureMode === "mach" ? "position" : "mach";
    const sim = fresh(pureMode);
    sim.setTuning({ki_per_s2: 0});
    sim.setMode(otherMode);
    drive(sim);
    assert.notEqual(sim.getState().integral_velocity_mm_s, 0);
    const before = sim.getState();
    sim.setMode(pureMode);
    assert.equal(sim.getState().time_s, before.time_s);
    assert.equal(sim.getState().actual_L_mm, before.actual_L_mm);
    assert.equal(sim.getState().velocity_mm_s, before.velocity_mm_s);
    assert.equal(sim.getState().integral_velocity_mm_s, 0);
    assert.equal(sim.step().integral_velocity_mm_s, 0);
  }
});

test("pure-P tuning cannot command farther out of a position boundary", () => {
  const sim = fresh("mach", {max_speed_mm_s: 10000, mach_kp_per_s: 10000, mach_ki_per_s2: 0});
  sim.setCommand(sim.getState().mach_max);
  sim.step(5);
  assert.equal(sim.getState().actual_L_mm, sim.getState().L_min_mm);
  assert.ok(sim.getState().equivalent_control_error_mm < 0);
  const physical = physicalState(sim);
  sim.setTuning({kp_per_s: 9000});
  assert.deepEqual(physicalState(sim), physical);
  assert.equal(sim.getState().velocity_command_mm_s, 0);
  assert.equal(sim.getState().position_limited, true);
  assert.equal(sim.getState().integral_velocity_mm_s, 0);
});
