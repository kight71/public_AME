"""Unit tests for thesis-style landing tracking reward (eq. 4-22)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_PLANNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/ame_locomotion/ame_locomotion/tasks/manager_based/ame_locomotion/mdp/planner.py"
)


def _load_planner():
    spec = importlib.util.spec_from_file_location("planner_landing_test", _PLANNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def planner():
    return _load_planner()


def test_perfect_contact_reward(planner):
    foot = torch.tensor([[[1.0, 0.0, 0.8], [0.0, -0.1, 0.8]]])
    target = foot.clone()
    contact = torch.tensor([[1.0, 1.0]])
    out = planner.landing_tracking_exp(foot, target, contact, sigma=0.12)
    assert out.item() == pytest.approx(2.0, abs=1e-5)


def test_no_contact_zero(planner):
    foot = torch.tensor([[[1.0, 0.0, 0.8], [0.0, -0.1, 0.8]]])
    target = torch.zeros_like(foot)
    contact = torch.tensor([[0.0, 0.0]])
    out = planner.landing_tracking_exp(foot, target, contact, sigma=0.12)
    assert out.item() == pytest.approx(0.0, abs=1e-5)


def test_offset_decays_with_sigma(planner):
    foot = torch.tensor([[[0.12, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    target = torch.zeros(1, 2, 3)
    contact = torch.tensor([[1.0, 0.0]])
    r_tight = planner.landing_tracking_exp(foot, target, contact, sigma=0.12).item()
    r_loose = planner.landing_tracking_exp(foot, target, contact, sigma=0.24).item()
    assert r_loose > r_tight
    assert r_tight == pytest.approx(0.367879, abs=1e-4)  # exp(-1)
