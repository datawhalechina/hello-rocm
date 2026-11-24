"""Teaching-friendly Minesweeper environment for the toy-cli workshop.

The file is intentionally small enough to read during an offline workshop. It
keeps the game state, Agent observation, candidate move generation, and tool
execution in one place so learners can see how an Agent connects to an
environment.
"""

from __future__ import annotations

import random
import string
from typing import Any

Coord = tuple[int, int]
ToolResult = dict[str, Any]


def format_coord(row: int, col: int) -> str:
    return f"{string.ascii_uppercase[row]}{col + 1}"


def parse_coord(raw: Any, board_size: int) -> Coord:
    """Accept flexible coordinates because LLM outputs are not always identical.

    Teaching point: a classroom Agent should tolerate common valid shapes such
    as "A1", {"row": "A", "col": 1}, or [0, 1], then normalize them before
    touching the environment.
    """

    if isinstance(raw, dict):
        row_raw, col_raw = raw.get("row"), raw.get("col")
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        row_raw, col_raw = raw
    elif isinstance(raw, str):
        text = raw.strip().upper()
        if not text or not text[0].isalpha():
            raise ValueError(f"cannot parse coordinate: {raw!r}")
        row_raw, col_raw = text[0], text[1:]
    else:
        raise ValueError(f"cannot parse coordinate: {raw!r}")

    if isinstance(row_raw, str):
        row_text = row_raw.strip().upper()
        if len(row_text) != 1 or not row_text.isalpha():
            raise ValueError(f"row must be one letter, got {row_raw!r}")
        row = ord(row_text) - ord("A")
    else:
        row = int(row_raw)

    if isinstance(col_raw, str):
        col_text = col_raw.strip()
        if not col_text.isdigit():
            raise ValueError(f"column must be an integer, got {col_raw!r}")
        col = int(col_text) - 1
    else:
        col = int(col_raw) - 1

    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(f"coordinate out of bounds for {board_size}x{board_size}: {raw!r}")
    return row, col


class MinesweeperGame:
    """Minimal game state used as the Agent's external world.

    Teaching point: the Agent does not "imagine" progress. It must read and
    modify a concrete state object, then inspect feedback after each tool call.
    """

    def __init__(
        self,
        board_size: int = 4,
        mine_count: int = 3,
        seed: int | None = 7,
        first_click_safe: bool = True,
    ) -> None:
        if not 2 <= board_size <= 9:
            raise ValueError("board_size must be between 2 and 9 for this notebook demo")
        if not 1 <= mine_count < board_size * board_size:
            raise ValueError("mine_count must be smaller than the board area")

        self.board_size = board_size
        self.mine_count = mine_count
        self.first_click_safe = first_click_safe
        self._rng = random.Random(seed)
        self.mines = [[False] * board_size for _ in range(board_size)]
        self.revealed = [[False] * board_size for _ in range(board_size)]
        self.flagged = [[False] * board_size for _ in range(board_size)]
        self.status = "playing"
        self.turn = 0
        self.first_click_done = False
        self.exploded_at: Coord | None = None
        self.history: list[dict[str, Any]] = []
        self._place_mines()

    def _place_mines(self) -> None:
        cells = [(r, c) for r in range(self.board_size) for c in range(self.board_size)]
        for row, col in self._rng.sample(cells, self.mine_count):
            self.mines[row][col] = True

    def _replace_mine_after_first_click(self, safe: Coord) -> None:
        # Workshop safety rail: the first click should open the game instead of
        # ending the demo immediately. This keeps attention on the Agent loop.
        row, col = safe
        if not self.mines[row][col]:
            return
        self.mines[row][col] = False
        for candidate_row in range(self.board_size):
            for candidate_col in range(self.board_size):
                if not self.mines[candidate_row][candidate_col] and (candidate_row, candidate_col) != safe:
                    self.mines[candidate_row][candidate_col] = True
                    return

    def neighbors(self, row: int, col: int) -> list[Coord]:
        out: list[Coord] = []
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                if d_row == 0 and d_col == 0:
                    continue
                next_row, next_col = row + d_row, col + d_col
                if 0 <= next_row < self.board_size and 0 <= next_col < self.board_size:
                    out.append((next_row, next_col))
        return out

    def neighbor_mine_count(self, row: int, col: int) -> int:
        return sum(1 for next_row, next_col in self.neighbors(row, col) if self.mines[next_row][next_col])

    def reveal(self, row: int, col: int) -> ToolResult:
        """Tool: reveal one cell and return environment feedback."""

        if self.status != "playing":
            return {"success": False, "message": f"game already {self.status}"}
        if self.flagged[row][col]:
            return {"success": False, "message": f"{format_coord(row, col)} is flagged; unflag first"}
        if self.revealed[row][col]:
            return {"success": False, "message": f"{format_coord(row, col)} already revealed"}

        if self.first_click_safe and not self.first_click_done:
            self._replace_mine_after_first_click((row, col))
        self.first_click_done = True
        self.turn += 1

        if self.mines[row][col]:
            self.revealed[row][col] = True
            self.exploded_at = (row, col)
            self.status = "lost"
            result = {"success": True, "message": f"BOOM at {format_coord(row, col)}", "outcome": "lost"}
            self._record("reveal", row, col, result)
            return result

        opened = self._flood_reveal(row, col)
        if self._check_won():
            self.status = "won"
        result = {
            "success": True,
            "message": f"revealed {len(opened)} cell(s) from {format_coord(row, col)}",
            "outcome": self.status,
        }
        self._record("reveal", row, col, result)
        return result

    def _flood_reveal(self, row: int, col: int) -> list[Coord]:
        opened: list[Coord] = []
        stack = [(row, col)]
        while stack:
            current_row, current_col = stack.pop()
            if self.revealed[current_row][current_col] or self.flagged[current_row][current_col]:
                continue
            if self.mines[current_row][current_col]:
                continue
            self.revealed[current_row][current_col] = True
            opened.append((current_row, current_col))
            if self.neighbor_mine_count(current_row, current_col) == 0:
                stack.extend(self.neighbors(current_row, current_col))
        return opened

    def flag(self, row: int, col: int) -> ToolResult:
        """Tool: mark one hidden cell as a suspected mine."""

        if self.status != "playing":
            return {"success": False, "message": f"game already {self.status}"}
        if self.revealed[row][col]:
            return {"success": False, "message": f"{format_coord(row, col)} already revealed; cannot flag"}
        if self.flagged[row][col]:
            return {"success": False, "message": f"{format_coord(row, col)} already flagged"}
        self.flagged[row][col] = True
        self.turn += 1
        result = {"success": True, "message": f"flagged {format_coord(row, col)}", "outcome": self.status}
        self._record("flag", row, col, result)
        return result

    def unflag(self, row: int, col: int) -> ToolResult:
        """Tool: remove one flag when the Agent changes its mind."""

        if self.status != "playing":
            return {"success": False, "message": f"game already {self.status}"}
        if not self.flagged[row][col]:
            return {"success": False, "message": f"{format_coord(row, col)} is not flagged"}
        self.flagged[row][col] = False
        self.turn += 1
        result = {"success": True, "message": f"unflagged {format_coord(row, col)}", "outcome": self.status}
        self._record("unflag", row, col, result)
        return result

    def _record(self, action: str, row: int, col: int, result: ToolResult) -> None:
        self.history.append(
            {
                "turn": self.turn,
                "action": action,
                "cell": format_coord(row, col),
                "result": result,
            }
        )

    def _check_won(self) -> bool:
        return all(
            self.mines[row][col] or self.revealed[row][col]
            for row in range(self.board_size)
            for col in range(self.board_size)
        )


def render_ascii(game: MinesweeperGame, reveal_mines: bool = False) -> str:
    header = "     " + "  ".join(f"{col + 1:>2}" for col in range(game.board_size))
    lines = [header]
    for row in range(game.board_size):
        glyphs = []
        for col in range(game.board_size):
            if game.revealed[row][col]:
                glyph = "*" if game.mines[row][col] else str(game.neighbor_mine_count(row, col)) or "."
                if glyph == "0":
                    glyph = "."
            elif reveal_mines and game.mines[row][col]:
                glyph = "*"
            elif game.flagged[row][col]:
                glyph = "F"
            else:
                glyph = "_"
            glyphs.append(f"{glyph:>2}")
        lines.append(f"  {string.ascii_uppercase[row]}  " + "  ".join(glyphs))
    return "\n".join(lines)


def compute_score(game: MinesweeperGame) -> dict[str, int | str]:
    revealed_safe = sum(
        1
        for row in range(game.board_size)
        for col in range(game.board_size)
        if game.revealed[row][col] and not game.mines[row][col]
    )
    correct_flags = sum(
        1
        for row in range(game.board_size)
        for col in range(game.board_size)
        if game.flagged[row][col] and game.mines[row][col]
    )
    wrong_flags = sum(
        1
        for row in range(game.board_size)
        for col in range(game.board_size)
        if game.flagged[row][col] and not game.mines[row][col]
    )
    safe_total = game.board_size * game.board_size - game.mine_count
    score = revealed_safe * 10 + correct_flags * 5 - wrong_flags * 5
    if game.status == "won":
        score += 50
    return {
        "score": score,
        "revealed_safe": revealed_safe,
        "safe_total": safe_total,
        "correct_flags": correct_flags,
        "wrong_flags": wrong_flags,
        "turn": game.turn,
        "status": game.status,
    }


def observe_for_llm(game: MinesweeperGame) -> dict[str, Any]:
    """Convert game state into the compact observation shown to the Agent.

    Teaching point: an Agent needs a deliberate observation format. Passing the
    entire Python object would be noisy; passing only board, score, allowed
    actions, and last feedback makes the next decision easier to explain.
    """

    return {
        "board": render_ascii(game),
        "state": compute_score(game),
        "allowed_actions": ["reveal", "flag", "unflag"],
        "last_action": game.history[-1] if game.history else None,
    }


def propose_minesweeper_actions(game: MinesweeperGame) -> list[dict[str, Any]]:
    """Generate legal candidate moves before asking the local model to choose.

    Teaching point: small local models are more reliable when we turn open-ended
    generation into selection from a short, valid candidate list.
    """

    if game.status != "playing":
        return []

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()

    def add(action: str, row: int, col: int, score: float, reason: str) -> None:
        key = (action, row, col)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "action": action,
                "args": {"row": string.ascii_uppercase[row], "col": col + 1},
                "score": score,
                "reason": reason,
            }
        )

    for row in range(game.board_size):
        for col in range(game.board_size):
            if not game.revealed[row][col] or game.mines[row][col]:
                continue
            number = game.neighbor_mine_count(row, col)
            if number == 0:
                continue
            hidden = [
                (next_row, next_col)
                for next_row, next_col in game.neighbors(row, col)
                if not game.revealed[next_row][next_col] and not game.flagged[next_row][next_col]
            ]
            flagged = [
                (next_row, next_col)
                for next_row, next_col in game.neighbors(row, col)
                if game.flagged[next_row][next_col]
            ]
            if hidden and number - len(flagged) == len(hidden):
                # Minesweeper rule: all remaining hidden neighbors must be mines.
                for next_row, next_col in hidden:
                    add(
                        "flag",
                        next_row,
                        next_col,
                        0.95,
                        f"{format_coord(row, col)} shows {number}; all hidden neighbors are mines.",
                    )
            if hidden and len(flagged) == number:
                # Minesweeper rule: once all mines around a number are flagged,
                # the other hidden neighbors are safe to reveal.
                for next_row, next_col in hidden:
                    add(
                        "reveal",
                        next_row,
                        next_col,
                        0.90,
                        f"{format_coord(row, col)} already has {number} flagged neighbor(s); this cell is safe.",
                    )

    if candidates:
        return sorted(candidates, key=lambda item: item["score"], reverse=True)[:6]

    hidden_cells = [
        (row, col)
        for row in range(game.board_size)
        for col in range(game.board_size)
        if not game.revealed[row][col] and not game.flagged[row][col]
    ]
    hidden_cells.sort(
        key=lambda cell: (
            not (cell[0] in (0, game.board_size - 1) and cell[1] in (0, game.board_size - 1)),
            not (cell[0] in (0, game.board_size - 1) or cell[1] in (0, game.board_size - 1)),
            cell,
        )
    )
    for row, col in hidden_cells[:3]:
        add("reveal", row, col, 0.50, "No certain move yet; choose a corner/edge as a simple fallback.")
    return candidates


def apply_decision(game: MinesweeperGame, decision: dict[str, Any]) -> tuple[MinesweeperGame, ToolResult]:
    """Dispatch one Agent JSON decision to a real game tool.

    Teaching point: this is where text becomes action. The Agent's JSON is not
    trusted blindly; it is checked, normalized, then mapped to a Python method.
    """

    action = decision.get("action")
    args = decision.get("args") or {}
    if action not in {"reveal", "flag", "unflag"}:
        return game, {"success": False, "message": f"unknown action: {action!r}", "outcome": game.status}
    try:
        row, col = parse_coord(args, game.board_size)
    except (TypeError, ValueError) as exc:
        return game, {"success": False, "message": f"bad coordinate: {exc}", "outcome": game.status}
    return game, getattr(game, action)(row, col)
