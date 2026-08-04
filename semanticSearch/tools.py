import ast
import operator
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from retrieval import retrieve_query

DATE_FORMAT = "%Y-%m-%d"

_ARITHMETIC = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Guards against 9**9**9 style expressions freezing the process
MAX_EXPONENT = 64
MAX_OPERAND = 10 ** 12


def ok(output: str, hits: list[dict] = None) -> dict:
    return {
        "ok": True,
        "output": output,
        "hits": hits or [],
    }


def failed(output: str) -> dict:
    return {
        "ok": False,
        "output": output,
        "hits": [],
    }


def search_documents(tool_input: str, context: dict) -> dict:
    """
    The whole retrieval pipeline as a tool: query expansion, hybrid search,
    RRF fusion, reranking and compression.
    """

    query = tool_input.strip()

    if not query:
        return failed("empty query")

    hits, trace = retrieve_query(query, context.get("transcript", ""))

    context["llm_calls"] = context.get("llm_calls", 0) + trace["llm_calls"]

    if not hits:
        return failed(f"no documents matched {query!r}")

    summary = "; ".join(
        f"{hit['document']} p.{hit['page']}"
        for hit in hits
    )

    return ok(f"{len(hits)} excerpt(s) retrieved: {summary}", hits)


def current_time(tool_input: str, context: dict) -> dict:
    """
    Input: an optional IANA timezone such as Asia/Kolkata. Empty means the
    machine's local time.
    """

    zone_name = tool_input.strip()

    if not zone_name:
        now = datetime.now()

        return ok(f"local time is {now.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        now = datetime.now(ZoneInfo(zone_name))
    except (ZoneInfoNotFoundError, ValueError):
        return failed(f"unknown timezone {zone_name!r}")

    return ok(f"time in {zone_name} is {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")


def date_math(tool_input: str, context: dict) -> dict:
    """
    Input is one of:
      diff YYYY-MM-DD YYYY-MM-DD   days between two dates
      add YYYY-MM-DD N             the date N days later
      sub YYYY-MM-DD N             the date N days earlier
    """

    parts = tool_input.strip().split()

    if len(parts) != 3:
        return failed(
            "expected `diff <date> <date>`, `add <date> <days>` or "
            "`sub <date> <days>`"
        )

    action, first, second = parts[0].lower(), parts[1], parts[2]

    try:
        start = datetime.strptime(first, DATE_FORMAT).date()
    except ValueError:
        return failed(f"{first!r} is not a YYYY-MM-DD date")

    if action == "diff":
        try:
            end = datetime.strptime(second, DATE_FORMAT).date()
        except ValueError:
            return failed(f"{second!r} is not a YYYY-MM-DD date")

        days = (end - start).days

        return ok(f"{days} day(s) from {first} to {second}")

    if action not in {"add", "sub"}:
        return failed(f"unknown action {action!r}, use diff, add or sub")

    try:
        days = int(second)
    except ValueError:
        return failed(f"{second!r} is not a whole number of days")

    if action == "sub":
        days = -days

    result = start + timedelta(days=days)

    return ok(f"{first} {action} {abs(days)} day(s) is {result.isoformat()}")


def calculator(tool_input: str, context: dict) -> dict:
    """
    Input is an arithmetic expression such as (1.5 * 12) - 4.

    Parsed into an AST and walked by hand. eval() would hand the model
    arbitrary code execution over its own output.
    """

    expression = tool_input.strip()

    if not expression:
        return failed("empty expression")

    try:
        tree = ast.parse(expression, mode="eval")
        value = _evaluate(tree.body)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as error:
        return failed(f"cannot evaluate {expression!r}: {error}")

    if isinstance(value, float) and value.is_integer():
        value = int(value)

    return ok(f"{expression} = {value}")


def _evaluate(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numbers are allowed")

        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITHMETIC:
        return _ARITHMETIC[type(node.op)](_evaluate(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC:
        left = _evaluate(node.left)
        right = _evaluate(node.right)

        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT or abs(left) > MAX_OPERAND:
                raise ValueError("exponent too large")

        return _ARITHMETIC[type(node.op)](left, right)

    raise ValueError("unsupported expression")


SEARCH_TOOL = "search_documents"

TOOLS = {
    SEARCH_TOOL: {
        "run": search_documents,
        "description": (
            "Look something up in the indexed documents. This is the only "
            "way to learn what the documents actually say."
        ),
        "input_format": "a standalone search query",
        "example": "annual leave carry over limit",
    },
    "current_time": {
        "run": current_time,
        "description": "The current date and time.",
        "input_format": "an IANA timezone, or empty for local time",
        "example": "Asia/Kolkata",
    },
    "date_math": {
        "run": date_math,
        "description": "Days between two dates, or a date shifted by N days.",
        "input_format": (
            "`diff YYYY-MM-DD YYYY-MM-DD`, `add YYYY-MM-DD N` or "
            "`sub YYYY-MM-DD N`"
        ),
        "example": "diff 2026-01-01 2026-08-04",
    },
    "calculator": {
        "run": calculator,
        "description": "Arithmetic. Use it instead of doing sums yourself.",
        "input_format": "an expression using + - * / // % ** and brackets",
        "example": "(1.5 * 12) - 4",
    },
}


def tool_names() -> list[str]:
    return list(TOOLS)


def describe_tools() -> str:
    """
    The catalog handed to the planner.
    """

    lines = []

    for name, tool in TOOLS.items():
        lines.append(
            f"- {name}: {tool['description']}\n"
            f"  input: {tool['input_format']}\n"
            f"  example input: {tool['example']}"
        )

    return "\n".join(lines)


def run_tool(name: str, tool_input: str, context: dict) -> dict:
    tool = TOOLS.get(name)

    if not tool:
        return failed(f"unknown tool {name!r}")

    try:
        return tool["run"](tool_input, context)
    except Exception as error:
        # A broken tool should cost one step, not the whole answer.
        return failed(f"{name} raised {type(error).__name__}: {error}")
