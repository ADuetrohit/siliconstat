"""SPICE engineering-notation parsing and formatting."""

from __future__ import annotations

import math

import pytest

from siliconstat.core.exceptions import NetlistSyntaxError
from siliconstat.core.units import format_eng, parse_value, parse_value_or_percent


@pytest.mark.parametrize("text,expected", [
    ("0", 0.0),
    ("5", 5.0),
    ("1.8", 1.8),
    (".5", 0.5),
    ("-2.5", -2.5),
    ("+3", 3.0),
    ("2.5e-3", 2.5e-3),
    ("1E6", 1e6),
    ("1T", 1e12),
    ("2G", 2e9),
    ("3k", 3e3),
    ("3K", 3e3),
    ("10k", 1e4),
    ("180n", 180e-9),
    ("5u", 5e-6),
    ("1p", 1e-12),
    ("2f", 2e-15),
    ("4a", 4e-18),
])
def test_basic_suffixes(text, expected):
    assert parse_value(text) == pytest.approx(expected, rel=1e-12)


def test_m_means_milli_and_meg_means_mega():
    """The classic SPICE trap: M is milli, MEG is mega."""
    assert parse_value("1M") == pytest.approx(1e-3)
    assert parse_value("1m") == pytest.approx(1e-3)
    assert parse_value("1MEG") == pytest.approx(1e6)
    assert parse_value("1meg") == pytest.approx(1e6)
    assert parse_value("1X") == pytest.approx(1e6)


def test_mil_is_matched_before_milli():
    assert parse_value("1mil") == pytest.approx(25.4e-6)
    # ... and plain 'm' still means milli, so the longest-match rule works.
    assert parse_value("1m") == pytest.approx(1e-3)


def test_micro_sign():
    assert parse_value("5µ") == pytest.approx(5e-6)


def test_trailing_unit_annotation_is_ignored():
    assert parse_value("10kOhm") == pytest.approx(1e4)
    assert parse_value("1.8V") == pytest.approx(1.8)
    assert parse_value("180nm") == pytest.approx(180e-9)
    assert parse_value("50fF") == pytest.approx(50e-15)


def test_embedded_whitespace_is_rejected():
    """A value token is a single lexical unit; '50f F' is two tokens."""
    with pytest.raises(NetlistSyntaxError):
        parse_value("50f F")


@pytest.mark.parametrize("text", ["", "   ", "abc", "1..2", "--3", "1.0.0", "k10"])
def test_malformed_values_raise(text):
    with pytest.raises(NetlistSyntaxError):
        parse_value(text)


def test_error_carries_line_information():
    with pytest.raises(NetlistSyntaxError) as excinfo:
        parse_value("bogus", line_no=17, source="demo.net")
    message = str(excinfo.value)
    assert "17" in message
    assert "demo.net" in message


def test_percentage_parsing():
    value, is_percent = parse_value_or_percent("2%")
    assert is_percent is True
    assert value == pytest.approx(0.02)

    value, is_percent = parse_value_or_percent("10k")
    assert is_percent is False
    assert value == pytest.approx(1e4)


def test_percentage_in_plain_position_is_rejected():
    with pytest.raises(NetlistSyntaxError):
        parse_value("2%")


def test_bad_percentage_raises():
    with pytest.raises(NetlistSyntaxError):
        parse_value_or_percent("abc%")


@pytest.mark.parametrize("value,unit,expected_prefix", [
    (1.234e-3, "A", "m"),
    (1.234e-6, "A", "u"),
    (1.234e-9, "F", "n"),
    (1.234e-12, "F", "p"),
    (1.234e3, "ohm", "k"),
    (1.234e6, "Hz", "M"),
    (1.234e9, "Hz", "G"),
    (1.234, "V", ""),
])
def test_format_eng_picks_the_right_prefix(value, unit, expected_prefix):
    text = format_eng(value, unit)
    assert text.endswith(f"{expected_prefix}{unit}")
    assert text.startswith("1.234")


def test_format_eng_edge_cases():
    assert format_eng(0.0, "V") == "0 V"
    assert "nan" in format_eng(float("nan"), "V")
    assert "inf" in format_eng(float("inf"), "V")
    assert "-inf" in format_eng(float("-inf"), "V")


def test_display_format_uses_the_si_convention():
    """Humans read M as mega, so display output must too."""
    assert format_eng(24.5e6, "Hz") == "24.5 MHz"
    assert format_eng(1.25e-3, "A") == "1.25 mA"


def test_spice_format_round_trips_through_the_parser():
    """spice=True output must be re-readable, including the M/MEG trap."""
    for value in (1.25e-3, 3.75, 6.25e-3, 125e3, 24.5e6, 812e-6, 1e12, 2e-15):
        text = format_eng(value, spice=True)
        assert parse_value(text) == pytest.approx(value, rel=1e-3), text


def test_display_format_of_mega_is_deliberately_not_parseable_as_mega():
    """Documents the SI-vs-SPICE asymmetry that spice=True exists to avoid."""
    assert parse_value(format_eng(24.5e6, spice=True)) == pytest.approx(24.5e6)
    # The display form says 'M', which SPICE reads as milli -- hence the flag.
    assert parse_value("24.5M") == pytest.approx(24.5e-3)


def test_negative_values_keep_their_sign():
    assert parse_value("-10u") == pytest.approx(-1e-5)
    assert format_eng(-1e-5, "A").startswith("-10")
