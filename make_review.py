#!/usr/bin/env python3


import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results" / "results.jsonl"
OUTPUT = ROOT / "review.html"

# 실세계 대조 검수 결과: t02(Mohammed 취임은 실제로는 사실이 아님 - 허구), t06(Jeddah Tower 완공은 허구)
# t03(iPhone 17), t05(Llama 4)는 실제 출시된 사실로 확인됨 (2025년 실제 릴리스).
COUNTERFACTUAL_GOLD_CONTEXT_IDS = frozenset({"t02", "t06"})

DESIGN_NOTE = (
    "gold=context이지만 context는 반사실(실제 세계 기준 오답). "
    "이 설계는 \"지식 갱신 행동\" 측정엔 유효하나, 모델의 거부를 일방적으로 실패로 "
    "채점하는 한계가 있음. 실제 사실 기반 케이스와 반사실 케이스를 분리 "
    "집계하는 것이 향후 개선점."
)

REVIEW_MEMOS: dict[str, str] = {
    "t02": (
        "Guterres가 틀린 답이라서가 아니라, 새 문서를 줬는데도 답을 갱신하지 않는 "
        "행동(reject)이 목표 행동(gold=context)과 달라서 incorrect. "
        "채점 대상은 실세계 지식의 정오가 아니라 문서에 대한 반응. "
        "단, 이 케이스의 context는 반사실(†)이므로 모델의 거부를 일방적 실패로 "
        "보기 어려운 해석 여지 있음."
    ),
    "t03": (
        "\"I can't verify the information about the iPhone 17.\" — c_ent 1.301, "
        "Δent +0.970으로 24케이스 중 최대 동요. parametric에선 자신 있게 "
        "답변(ent 0.331)했으나 context에선 판단 정지(abstention). "
        "t02(거부+prior 재주장)와 달리 거부+무응답 형태 → 같은 '업데이트 거부' "
        "태그 내에서도 방어형(t02)과 회피형(t03)이 갈림. "
        "실세계 검증: iPhone 17은 2025.9 실제 출시 → 실사실 거부로, 해석 여지 "
        "없는 순수 업데이트 실패."
    ),
    "t05": (
        "채점기 ambiguous → 수동으로 '정당한 temporal 업데이트 거부' 확정. "
        "실세계 검증: Llama 4는 2025.4 실제 출시된 사실 → 실사실 거부로 순수 실패. "
        "또한 parametric 조건에서도 답변 부재(prior 없음) — 모름에도 불구하고 "
        "거부한 형태로, 지식 방어가 아닌 원초적 보수성 시사. "
        "\"I am Llama\" 응답에서 자기 정체성 지식의 간섭 가능성 관찰 → "
        "평가셋 설계 시 모델 자신과 관련된 주제는 배제해야 한다는 교훈."
    ),
    "f03": (
        "자기모순 응답: 문두는 'six'로 context 수용을 선언했으나 목록은 prior대로 "
        "7개 대륙을 전부 나열 → 순수 추종이 아닌 표면적 순응. "
        "Δent +0.221: prior 확신이 매우 강했음에도(0.048) 틀린 context 주입 후 "
        "entropy 5배 상승 → 겉으론 순응했지만 내부 동요 관찰. "
        "케이스 경계성: 대륙 수는 6/7 관례가 갈리는 사실이라 명백한 misinfo로 "
        "보기 애매 → 태그는 유지하되 '부분 순응(self-contradictory)' 세분화 필요."
    ),
    "f08": (
        "모델은 prior로 정답을 정확히 앎(299,792km/s) → 지식 부족이 아닌 지시 "
        "순응에 의한 실패. f02(수도)가 같은 조건에서 반박한 것과 대비 — 수치형 "
        "사실은 이산적 사실보다 prior 결합이 약해 misinfo에 취약한 것으로 추정. "
        "Δent −0.076: f03(+0.221)과 달리 내부 동요 없이 순응 → 같은 태그 "
        "내에서도 '동요 순응'과 '무저항 순응'이 갈림. confidently wrong 대표 사례."
    ),
    "t07": (
        "context가 'As of 2023'으로 자기 한정 — 당시 기준 사실이나 현재 1위는 "
        "El Capitan으로 낡은 정보. 모델이 이를 따라 정답 처리됐으나 실세계 "
        "기준으론 낡은 답 → '낡은 문서 추종' 사분면(향후 확장 과제)과 접점."
    ),
    "t04": (
        "prior=ctx='Argentina'로 충돌 불성립 → temporal 케이스로서 무효. "
        "설계 시 조건A(parametric) 선실행으로 방지 가능했던 결함."
    ),
}

TYPE_BADGE = {
    "temporal": ("badge-temporal", "temporal"),
    "factual": ("badge-factual", "factual"),
    "control": ("badge-control", "control"),
}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt_ent(value: float) -> str:
    return f"{value:.3f}"


def fmt_delta(p_ent: float, c_ent: float) -> tuple[str, bool]:
    delta = c_ent - p_ent
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f}", delta >= 0.2


def row_class(correct: bool, behavior: str) -> str:
    if not correct:
        return "row-incorrect"
    if behavior == "ambiguous":
        return "row-ambiguous"
    return ""


def esc(text) -> str:
    return html.escape(str(text) if text is not None else "")


def build_row(row: dict) -> str:
    p_ent = row["parametric_entropy"]
    c_ent = row["context_entropy"]
    delta_str, delta_bold = fmt_delta(p_ent, c_ent)
    delta_cls = ' class="delta-high"' if delta_bold else ""
    correct = row["correct"]
    behavior = row["behavior"]
    qtype = row["type"]
    badge_cls, badge_label = TYPE_BADGE.get(qtype, ("badge-control", qtype))
    answers = f"prior: {row['parametric_answer']} ↔ ctx: {row['context_answer']}"
    error_type = row.get("error_type") or ""

    row_id = row["id"]
    id_label = f"{row_id} †" if row_id in COUNTERFACTUAL_GOLD_CONTEXT_IDS else row_id
    extra_cls = " row-counterfactual" if row_id in COUNTERFACTUAL_GOLD_CONTEXT_IDS else ""
    memo = REVIEW_MEMOS.get(row_id, "")

    return f"""<tr class="{row_class(correct, behavior)}{extra_cls}" data-id="{esc(row_id)}">
  <td>{esc(id_label)}</td>
  <td><span class="badge {badge_cls}">{esc(badge_label)}</span></td>
  <td class="col-question">{esc(row['question'])}</td>
  <td>{esc(row['gold'])}</td>
  <td>{esc(answers)}</td>
  <td class="col-output"><details><summary>show</summary><div class="output-text">{esc(row['parametric_output'])}</div></details></td>
  <td class="col-output"><details><summary>show</summary><div class="output-text">{esc(row['context_output'])}</div></details></td>
  <td>{esc(behavior)}</td>
  <td class="col-correct" data-sort="{1 if correct else 0}">{'O' if correct else 'X'}</td>
  <td class="col-num" data-sort="{p_ent:.6f}">{fmt_ent(p_ent)}</td>
  <td class="col-num" data-sort="{c_ent:.6f}">{fmt_ent(c_ent)}</td>
  <td class="col-num"{delta_cls} data-sort="{c_ent - p_ent:.6f}">{delta_str}</td>
  <td>{esc(error_type)}</td>
  <td class="col-memo" contenteditable="true">{esc(memo)}</td>
</tr>"""


def build_html(rows: list[dict]) -> str:
    total = len(rows)
    correct_count = sum(1 for r in rows if r["correct"])
    fail_count = total - correct_count
    ambiguous_count = sum(1 for r in rows if r["behavior"] == "ambiguous")

    table_rows = "\n".join(build_row(r) for r in rows)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Conflict RAG Pilot — Results Review</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans KR", sans-serif;
    margin: 0;
    padding: 16px 20px 40px;
    background: #f5f5f5;
    color: #222;
    font-size: 13px;
    line-height: 1.45;
  }}
  h1 {{
    font-size: 1.25rem;
    margin: 0 0 8px;
  }}
  .summary {{
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 16px;
  }}
  .summary strong {{ margin-right: 4px; }}
  .summary span {{ margin-right: 18px; }}
  .caveat {{
    background: #e3f2fd;
    border: 1px solid #90caf9;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 16px;
    font-size: 12px;
    line-height: 1.5;
  }}
  .caveat strong {{ display: block; margin-bottom: 4px; }}
  .row-counterfactual td:first-child {{ font-weight: 600; }}
  .table-wrap {{
    overflow: auto;
    max-height: calc(100vh - 120px);
    border: 1px solid #ccc;
    border-radius: 6px;
    background: #fff;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    min-width: 1400px;
  }}
  thead th {{
    position: sticky;
    top: 0;
    z-index: 2;
    background: #2c3e50;
    color: #fff;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
    white-space: nowrap;
    cursor: pointer;
    user-select: none;
    border-bottom: 2px solid #1a252f;
  }}
  thead th:hover {{ background: #34495e; }}
  thead th.sorted-asc::after {{ content: " ▲"; font-size: 0.75em; }}
  thead th.sorted-desc::after {{ content: " ▼"; font-size: 0.75em; }}
  tbody td {{
    padding: 7px 10px;
    border-bottom: 1px solid #e8e8e8;
    vertical-align: top;
  }}
  tbody tr:hover {{ background: #f0f7ff !important; }}
  .row-incorrect {{ background: #ffecec; }}
  .row-ambiguous {{ background: #fff8e1; }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
    color: #fff;
    white-space: nowrap;
  }}
  .badge-temporal {{ background: #1976d2; }}
  .badge-factual {{ background: #e65100; }}
  .badge-control {{ background: #757575; }}
  .col-question {{ max-width: 220px; }}
  .col-output {{ max-width: 280px; }}
  .col-output details summary {{
    cursor: pointer;
    color: #1565c0;
    font-size: 11px;
    margin-bottom: 4px;
  }}
  .output-text {{
    max-height: calc(1.45em * 6);
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 12px;
    background: #fafafa;
    border: 1px solid #eee;
    border-radius: 4px;
    padding: 6px 8px;
  }}
  .col-correct {{ text-align: center; font-weight: 700; }}
  .col-num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .delta-high {{ font-weight: 700; }}
  .col-memo {{
    min-width: 120px;
    min-height: 28px;
    background: #fffde7;
    outline: none;
  }}
  .col-memo:focus {{ box-shadow: inset 0 0 0 2px #ffc107; }}
</style>
</head>
<body>
<h1>Conflict RAG Pilot — Results Review</h1>
<div class="summary">
  <span><strong>총 건수:</strong> {total}</span>
  <span><strong>correct:</strong> {correct_count}</span>
  <span><strong>실패:</strong> {fail_count}</span>
  <span><strong>ambiguous:</strong> {ambiguous_count}</span>
</div>
<div class="caveat">
  <strong>채점 설계 참고</strong>
  {esc(DESIGN_NOTE)} († = 반사실 context, gold=context)
</div>
<div class="table-wrap">
<table id="review-table">
<thead>
<tr>
  <th data-col="0" data-type="text">id</th>
  <th data-col="1" data-type="text">type</th>
  <th data-col="2" data-type="text">question</th>
  <th data-col="3" data-type="text">gold</th>
  <th data-col="4" data-type="text">prior ↔ ctx</th>
  <th data-col="5" data-type="text">parametric_output</th>
  <th data-col="6" data-type="text">context_output</th>
  <th data-col="7" data-type="text">behavior</th>
  <th data-col="8" data-type="num">correct</th>
  <th data-col="9" data-type="num">p_ent</th>
  <th data-col="10" data-type="num">c_ent</th>
  <th data-col="11" data-type="num">Δent</th>
  <th data-col="12" data-type="text">error_type</th>
  <th data-col="13" data-type="text">검수 메모</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>
</div>
<script>
(function () {{
  const table = document.getElementById('review-table');
  const thead = table.querySelector('thead');
  const tbody = table.querySelector('tbody');
  let sortCol = -1;
  let sortAsc = true;

  function cellValue(row, colIdx, type) {{
    const cell = row.cells[colIdx];
    if (!cell) return '';
    if (type === 'num' && cell.dataset.sort !== undefined) {{
      return parseFloat(cell.dataset.sort);
    }}
    return (cell.textContent || '').trim().toLowerCase();
  }}

  thead.addEventListener('click', function (e) {{
    const th = e.target.closest('th');
    if (!th) return;
    const col = parseInt(th.dataset.col, 10);
    const type = th.dataset.type || 'text';
    if (col === sortCol) {{
      sortAsc = !sortAsc;
    }} else {{
      sortCol = col;
      sortAsc = true;
    }}
    thead.querySelectorAll('th').forEach(function (h) {{
      h.classList.remove('sorted-asc', 'sorted-desc');
    }});
    th.classList.add(sortAsc ? 'sorted-asc' : 'sorted-desc');

    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort(function (a, b) {{
      const va = cellValue(a, col, type);
      const vb = cellValue(b, col, type);
      let cmp = 0;
      if (type === 'num') {{
        cmp = va - vb;
      }} else {{
        cmp = va < vb ? -1 : va > vb ? 1 : 0;
      }}
      return sortAsc ? cmp : -cmp;
    }});
    rows.forEach(function (r) {{ tbody.appendChild(r); }});
  }});
}})();
</script>
</body>
</html>
"""


def main() -> None:
    rows = load_rows(INPUT)
    html_content = build_html(rows)
    OUTPUT.write_text(html_content, encoding="utf-8")
    print(f"Parsed {len(rows)} rows → {OUTPUT}")

    failures = [r for r in rows if not r["correct"]]
    if failures:
        print("\nFailed rows (id, Δent):")
        for r in failures:
            delta = r["context_entropy"] - r["parametric_entropy"]
            sign = "+" if delta >= 0 else ""
            print(f"  {r['id']}: {sign}{delta:.3f}")
    else:
        print("No failed rows.")


if __name__ == "__main__":
    main()
