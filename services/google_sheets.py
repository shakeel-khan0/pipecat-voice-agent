import os
from datetime import datetime
from time import perf_counter
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from loguru import logger

from services.google_auth import get_sheets_service
from trace_recorder import trace_recorder


load_dotenv()

SPREADSHEET_ID = os.getenv("GOOGLE_SHEET_ID")
SHEET_NAME = "Sheet1"
TARGET_RANGE = f"{SHEET_NAME}!A:K"


def _masked_spreadsheet_id():
    if not SPREADSHEET_ID:
        return "<missing>"
    if len(SPREADSHEET_ID) <= 8:
        return "<configured>"
    return f"{SPREADSHEET_ID[:4]}...{SPREADSHEET_ID[-4:]}"


def save_lead(
    name: str,
    email: str,
    business: str,
    industry: str,
    problem: str,
    volume: str,
    current_system: str,
    interested_service: str,
    meeting_date: str,
    meeting_time: str,
):
    started = perf_counter()
    if not SPREADSHEET_ID:
        trace_recorder.record(
            "external_operation",
            service="google_sheets",
            operation="values.append",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            success=False,
            range=TARGET_RANGE,
            error_type="ConfigurationError",
            application_retries=0,
        )
        return {
            "success": False,
            "error": "GOOGLE_SHEET_ID is not configured.",
        }

    try:
        service = get_sheets_service()
    except Exception as exc:
        trace_recorder.record(
            "external_operation",
            service="google_sheets",
            operation="values.append",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            success=False,
            range=TARGET_RANGE,
            error_type=type(exc).__name__,
            application_retries=0,
        )
        raise

    timestamp = datetime.now(
        ZoneInfo("Asia/Karachi")
    ).strftime("%Y-%m-%d %H:%M:%S")

    row = [[
        timestamp,
        name,
        email,
        business,
        industry,
        problem,
        volume,
        current_system,
        interested_service,
        meeting_date,
        meeting_time,
    ]]

    logger.warning(
        "Google Sheets append target: spreadsheet={} range={}",
        _masked_spreadsheet_id(),
        TARGET_RANGE,
    )

    try:
        result = service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range=TARGET_RANGE,
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            includeValuesInResponse=True,
            responseValueRenderOption="UNFORMATTED_VALUE",
            body={"values": row},
        ).execute()
    except Exception as exc:
        trace_recorder.record(
            "external_operation",
            service="google_sheets",
            operation="values.append",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            success=False,
            range=TARGET_RANGE,
            error_type=type(exc).__name__,
            application_retries=0,
        )
        raise

    updates = result.get("updates") or {}
    updated_range = updates.get("updatedRange")
    updated_rows = updates.get("updatedRows", 0)
    updated_columns = updates.get("updatedColumns", 0)
    updated_cells = updates.get("updatedCells", 0)
    returned_values = (updates.get("updatedData") or {}).get("values") or []

    logger.warning(
        "Google Sheets append result: range={} rows={} columns={} cells={}",
        updated_range or "<missing>",
        updated_rows,
        updated_columns,
        updated_cells,
    )

    write_confirmed = (
        bool(updated_range)
        and updated_rows == 1
        and updated_columns == len(row[0])
        and updated_cells == len(row[0])
        and len(returned_values) == 1
        and len(returned_values[0]) == len(row[0])
    )

    if not write_confirmed:
        logger.error("Google Sheets API did not confirm a complete lead row write.")
        trace_recorder.record(
            "external_operation",
            service="google_sheets",
            operation="values.append",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            success=False,
            range=TARGET_RANGE,
            updated_rows=updated_rows,
            updated_cells=updated_cells,
            error_type="WriteNotConfirmed",
            application_retries=0,
        )
        return {
            "success": False,
            "error": "Google Sheets did not confirm that the lead row was written.",
            "updated_range": updated_range,
            "updated_rows": updated_rows,
            "updated_cells": updated_cells,
        }

    trace_recorder.record(
        "external_operation",
        service="google_sheets",
        operation="values.append",
        duration_ms=round((perf_counter() - started) * 1000, 3),
        success=True,
        range=TARGET_RANGE,
        updated_rows=updated_rows,
        updated_cells=updated_cells,
        application_retries=0,
    )
    return {
        "success": True,
        "updated_range": updated_range,
        "updated_rows": updated_rows,
        "updated_cells": updated_cells,
    }

if __name__ == "__main__":
    print("Testing Google Sheets connection...")
