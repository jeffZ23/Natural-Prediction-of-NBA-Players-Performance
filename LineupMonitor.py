"""
Lineup and availability monitoring utilities.

This module wraps a couple of ESPN public endpoints so we can detect when a
player returns from an absence and pushes another player out of a starting or
major rotation spot. It also surfaces players that are currently sidelined,
which can unlock opportunity for teammates in predictive models.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

import requests

TEAM_ABBR_TO_ESPN_ID: Dict[str, str] = {
    "ATL": "1",
    "BOS": "2",
    "BKN": "3",
    "CHA": "4",
    "CHI": "5",
    "CLE": "6",
    "DAL": "7",
    "DEN": "8",
    "DET": "9",
    "GS": "10",
    "HOU": "11",
    "IND": "12",
    "LAC": "13",
    "LAL": "14",
    "MEM": "15",
    "MIA": "16",
    "MIL": "17",
    "MIN": "18",
    "NO": "19",
    "NY": "20",
    "OKC": "21",
    "ORL": "22",
    "PHI": "23",
    "PHX": "24",
    "POR": "25",
    "SAC": "26",
    "SA": "27",
    "TOR": "28",
    "UTA": "29",
    "WSH": "30",
}


class ESPNDataError(RuntimeError):
    """Raised when data cannot be retrieved from the ESPN API."""


def _fetch_json(url: str) -> Dict:
    """Fetch JSON with a browser-like user agent to avoid common blocking."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; lineup-monitor/1.0)"}

    if requests is not None:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as exc:  # pragma: no cover - network dependent
        raise ESPNDataError(f"Unable to fetch data from ESPN: {exc}") from exc


def get_espn_team_id(team_abbr: str) -> str:
    """Return the ESPN team identifier for a given NBA abbreviation."""
    abbr = team_abbr.upper()
    if abbr not in TEAM_ABBR_TO_ESPN_ID:
        raise ValueError(f"Unknown team abbreviation: {team_abbr}")
    return TEAM_ABBR_TO_ESPN_ID[abbr]


def _parse_depth_chart(data: Dict, team_abbr: str) -> pd.DataFrame:
    """Extract a normalized depth chart DataFrame from an ESPN team payload."""
    depth_charts = data.get("team", {}).get("depthCharts", [])
    rows = []

    for chart in depth_charts:
        position_group = chart.get("displayName") or chart.get("position", {}).get("displayName")
        for position in chart.get("positions", []):
            slot_label = position.get("displayName") or position.get("position", {}).get("displayName")
            players = position.get("players", [])
            for rank, player in enumerate(players, start=1):
                athlete = player.get("athlete") or player.get("player", {})
                status_info = athlete.get("status") or player.get("status", {})
                status_type = (status_info.get("type") or {}).get("name") if isinstance(status_info, dict) else None
                detail = (status_info.get("type") or {}).get("description") if isinstance(status_info, dict) else None

                rows.append(
                    {
                        "team": team_abbr.upper(),
                        "position_group": position_group,
                        "slot": slot_label,
                        "rank": rank,
                        "player": athlete.get("displayName") or athlete.get("fullName"),
                        "player_id": athlete.get("id"),
                        "status": status_type or "active",
                        "status_detail": detail,
                    }
                )

    return pd.DataFrame(rows)


def _parse_injuries(data: Dict, team_abbr: str) -> pd.DataFrame:
    """Normalize injury/availability information into a DataFrame."""
    injury_groups: Iterable = data.get("team", {}).get("injuries") or data.get("injuries", []) or []
    rows = []

    for entry in injury_groups:
        athlete = entry.get("athlete") or entry.get("player", {})
        status_info = entry.get("status") or entry.get("type", {})
        injury_status = None
        detail = None
        expected_return = None

        if isinstance(status_info, dict):
            injury_status = (status_info.get("type") or {}).get("name") or status_info.get("name")
            detail = (status_info.get("type") or {}).get("description") or status_info.get("detail")
            expected_return = status_info.get("expectedReturn") or status_info.get("expectedReturnDate")

        rows.append(
            {
                "team": team_abbr.upper(),
                "player": athlete.get("displayName") or athlete.get("fullName"),
                "player_id": athlete.get("id"),
                "status": injury_status or "unknown",
                "detail": detail,
                "expected_return": expected_return,
            }
        )

    return pd.DataFrame(rows)


def get_team_depth_chart(team_abbr: str) -> pd.DataFrame:
    """Download the latest depth chart for a team from ESPN."""
    team_id = get_espn_team_id(team_abbr)
    url = f"https://site.web.api.espn.com/apis/v2/sports/basketball/nba/teams/{team_id}?enable=roster,depthchart"
    data = _fetch_json(url)
    depth_chart_df = _parse_depth_chart(data, team_abbr)

    if depth_chart_df.empty:
        raise ESPNDataError(f"No depth chart data returned for {team_abbr}.")
    return depth_chart_df


def get_team_injuries(team_abbr: str) -> pd.DataFrame:
    """Download the current injury/availability report for a team from ESPN."""
    team_id = get_espn_team_id(team_abbr)
    url = f"https://site.web.api.espn.com/apis/v2/sports/basketball/nba/teams/{team_id}/injuries"
    data = _fetch_json(url)
    injuries_df = _parse_injuries(data, team_abbr)
    return injuries_df


def save_snapshot(df: pd.DataFrame, path: str) -> Path:
    """Persist a DataFrame snapshot to disk in JSON format."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path_obj, orient="records", indent=2)
    return path_obj


def load_snapshot(path: str) -> pd.DataFrame:
    """Load a DataFrame snapshot previously created with :func:`save_snapshot`."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")
    return pd.read_json(path_obj)


def detect_new_starters(current_depth: pd.DataFrame, previous_depth: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Identify starter changes (rank 1 for each position slot) between snapshots."""
    if previous_depth is None or previous_depth.empty:
        return pd.DataFrame()

    current_starters = current_depth[current_depth["rank"] == 1]
    previous_starters = previous_depth[previous_depth["rank"] == 1]

    merged = current_starters.merge(
        previous_starters,
        on=["team", "position_group", "slot", "rank"],
        how="left",
        suffixes=("_current", "_previous"),
    )
    return merged[merged["player_current"] != merged["player_previous"]]


def detect_returning_players(
    current_injuries: pd.DataFrame,
    previous_injuries: Optional[pd.DataFrame],
    current_depth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flag players who were recently sidelined but now appear active in the depth chart.

    This compares the new injury report to a previous snapshot. Players that were
    previously listed as anything other than "active" and are no longer flagged
    as such are returned.
    """
    if previous_injuries is None or previous_injuries.empty:
        return pd.DataFrame()

    inactive_previous = previous_injuries[previous_injuries["status"].str.lower() != "active"]
    active_now = set(current_depth["player"].dropna().unique())

    returning_records = []
    for _, row in inactive_previous.iterrows():
        if row["player"] in active_now:
            returning_records.append(row)

    return pd.DataFrame(returning_records)


def summarize_lineup_signals(
    team_abbr: str,
    previous_depth_snapshot: Optional[str] = None,
    previous_injury_snapshot: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Retrieve lineup context: active injuries, returning players, and starter swaps.

    Args:
        team_abbr: NBA abbreviation (e.g., "ATL").
        previous_depth_snapshot: Optional path to a saved depth chart snapshot.
        previous_injury_snapshot: Optional path to a saved injury snapshot.

    Returns:
        Dictionary of DataFrames keyed by signal type.
    """
    current_depth = get_team_depth_chart(team_abbr)
    current_injuries = get_team_injuries(team_abbr)

    prev_depth_df = load_snapshot(previous_depth_snapshot) if previous_depth_snapshot else None
    prev_injury_df = load_snapshot(previous_injury_snapshot) if previous_injury_snapshot else None

    return {
        "current_depth": current_depth,
        "current_injuries": current_injuries,
        "sidelined_players": current_injuries[current_injuries["status"].str.lower() != "active"],
        "returning_players": detect_returning_players(current_injuries, prev_injury_df, current_depth),
        "starter_changes": detect_new_starters(current_depth, prev_depth_df),
    }


def _print_table(title: str, df: pd.DataFrame) -> None:
    """Pretty-print a short DataFrame summary to stdout."""

    print(f"\n{title}")
    print("-" * len(title))
    if df is None or df.empty:
        print("(none)")
        return

    print(df.to_string(index=False))


def _default_snapshot_path(team: str, kind: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    return Path("snapshots") / f"{team.lower()}-{kind}-{timestamp}.json"


def main(argv: Optional[Iterable[str]] = None) -> None:
    """CLI entry point for smoke-testing lineup monitoring."""

    import argparse

    parser = argparse.ArgumentParser(description="Pull and compare ESPN lineup data for a team.")
    parser.add_argument("--team", required=True, help="NBA team abbreviation, e.g., ATL")
    parser.add_argument("--prev-depth", dest="prev_depth", help="Path to previous depth chart snapshot")
    parser.add_argument("--prev-injuries", dest="prev_injuries", help="Path to previous injuries snapshot")
    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        help="Save current depth and injury snapshots to the snapshots/ folder for later comparison.",
    )

    args = parser.parse_args(argv)

    if args.prev_depth:
        print(f"Using previous depth snapshot: {args.prev_depth}")
    if args.prev_injuries:
        print(f"Using previous injury snapshot: {args.prev_injuries}")

    signals = summarize_lineup_signals(
        args.team,
        previous_depth_snapshot=args.prev_depth,
        previous_injury_snapshot=args.prev_injuries,
    )

    if args.save_snapshots:
        depth_path = save_snapshot(signals["current_depth"], _default_snapshot_path(args.team, "depth"))
        injuries_path = save_snapshot(signals["current_injuries"], _default_snapshot_path(args.team, "injuries"))
        print(f"Saved current depth chart to {depth_path}")
        print(f"Saved current injury report to {injuries_path}")

    _print_table("Current depth chart (rank 1 = starters)", signals["current_depth"])
    _print_table("Current injury report", signals["current_injuries"])
    _print_table("Starter changes vs previous snapshot", signals["starter_changes"])
    _print_table("Returning players (were out, now active)", signals["returning_players"])
    _print_table("Sidelined players", signals["sidelined_players"])


if __name__ == "__main__":
    main()
