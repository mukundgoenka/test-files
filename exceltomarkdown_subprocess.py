"""Convert Excel workbooks into Markdown files by subprocess.

Usage:
    python exceltomarkdown_subprocess.py input.xlsx
    python exceltomarkdown_subprocess.py input.xlsx --output-dir excel-md-subprocess
    python exceltomarkdown_subprocess.py input.xlsx --sheet Sheet1 --subprocess-column SubProcess
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "pandas is required to run this script. Install it with `pip install pandas openpyxl`"
    ) from exc


def sanitize_filename(value: str) -> str:
    value = str(value).strip()
    if not value:
        return "no-subprocess"
    value = re.sub(r"[\\/\?%\*:|\"<>]", "-", value)
    value = re.sub(r"\s+", "_", value)
    return value


def find_column(columns: list[str], preferred: str) -> str | None:
    normalized = {col.strip().lower(): col for col in columns}
    return normalized.get(preferred.strip().lower())


def write_markdown_file(path: Path, df: pd.DataFrame, include_index: bool, na_rep: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if na_rep:
        df = df.fillna(na_rep)
    markdown_text = df.to_markdown(index=include_index, tablefmt="pipe")
    path.write_text(markdown_text, encoding="utf-8")
    return path


def excel_to_markdown_by_subprocess(
    input_path: Path,
    output_dir: Path | None = None,
    sheets: list[str] | None = None,
    include_index: bool = False,
    na_rep: str = "",
    subprocess_column: str = "SubProcess",
) -> list[Path]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Excel file not found: {input_path}")

    if sheets:
        sheet_names = [name.strip() for name in sheets if name.strip()]
    else:
        sheet_names = None

    excel_data = pd.read_excel(
        input_path,
        sheet_name=sheet_names,
        engine="openpyxl",
    )

    if output_dir is None:
        output_dir = input_path.with_suffix("")
        output_dir = output_dir.parent / f"{input_path.stem}_subprocess_md"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(excel_data, dict):
        sheet_items = excel_data.items()
    else:
        sheet_items = [(excel_data.name if hasattr(excel_data, "name") else "Sheet1", excel_data)]

    generated_files: list[Path] = []
    for sheet_name, df in sheet_items:
        if df.empty:
            continue

        sheet_dir = output_dir
        if len(sheet_items) > 1:
            sheet_dir = output_dir / sanitize_filename(sheet_name)
            sheet_dir.mkdir(parents=True, exist_ok=True)

        column_name = find_column(list(df.columns), subprocess_column)
        if column_name is not None:
            grouped = df.groupby(df[column_name].fillna("No SubProcess"), sort=False)
            for subprocess_value, group in grouped:
                if subprocess_value == "":
                    subprocess_value = "No SubProcess"
                file_name = sanitize_filename(subprocess_value)
                output_file = sheet_dir / f"{file_name}.md"
                generated_files.append(
                    write_markdown_file(output_file, group, include_index, na_rep)
                )
        else:
            output_file = sheet_dir / f"{sanitize_filename(sheet_name)}.md"
            generated_files.append(
                write_markdown_file(output_file, df, include_index, na_rep)
            )

    return generated_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an Excel workbook into Markdown files grouped by subprocess."
    )
    parser.add_argument(
        "input",
        metavar="INPUT",
        help="Path to the Excel file to convert (.xlsx, .xls).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory where generated Markdown files are written.",
    )
    parser.add_argument(
        "-s",
        "--sheet",
        help="Comma-separated sheet names to convert. Defaults to all sheets.",
    )
    parser.add_argument(
        "--subprocess-column",
        default="SubProcess",
        help="Column name used to group rows into separate Markdown files. Defaults to 'SubProcess'.",
    )
    parser.add_argument(
        "--include-index",
        action="store_true",
        help="Keep the DataFrame index in each Markdown output.",
    )
    parser.add_argument(
        "--na-rep",
        default="",
        help="String representation for missing values in the output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None
    sheets = args.sheet.split(",") if args.sheet else None

    generated_files = excel_to_markdown_by_subprocess(
        input_path=input_path,
        output_dir=output_dir,
        sheets=sheets,
        include_index=args.include_index,
        na_rep=args.na_rep,
        subprocess_column=args.subprocess_column,
    )

    if generated_files:
        print(f"Generated {len(generated_files)} Markdown file(s) in {output_dir or input_path.parent / f'{input_path.stem}_subprocess_md'}")
    else:
        print("No Markdown files were generated. The workbook may be empty.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
